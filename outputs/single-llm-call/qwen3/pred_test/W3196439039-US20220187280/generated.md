# DESCRIPTION

## FIELD

- define GEDI polypeptides

Genetically encoded death indicators (GEDI) are engineered polypeptide constructs designed to detect the irreversible loss of calcium homeostasis in living cells, particularly neurons, as a precise and unambiguous marker of cell death. These polypeptides consist of a modified calcium-binding domain derived from calmodulin or related calcium-sensing proteins, fused to a circularly permuted fluorescent protein that undergoes a conformational change upon calcium binding, resulting in a measurable increase in fluorescence intensity. The GEDI polypeptide is further fused via a self-cleaving P2a peptide sequence to a second, constitutively expressed fluorescent protein that serves as an internal reference for normalization. This design enables pseudo-ratiometric quantification of the GEDI signal relative to baseline expression levels, thereby minimizing artifacts arising from variations in transfection efficiency, photobleaching, or cellular volume changes. The calcium-binding domain of the GEDI polypeptide has been specifically modified to exhibit a dissociation constant (Kd) that is substantially higher than that of conventional genetically encoded calcium indicators, rendering it insensitive to physiological fluctuations in cytosolic calcium associated with neuronal activity but highly responsive to pathological elevations in intracellular calcium that occur upon plasma membrane rupture. The resulting polypeptide is expressed intracellularly under the control of a cell-type-specific promoter and remains quiescent during normal cellular function, only activating upon the catastrophic failure of calcium regulation that accompanies irreversible cell death. The GEDI polypeptide is thus a molecular sensor that translates the biophysical event of membrane integrity loss into a stable, quantifiable, and visually discernible fluorescent signal.

## BACKGROUND

- motivate calcium ions

Calcium ions serve as critical second messengers in nearly all eukaryotic cells, orchestrating a vast array of physiological processes ranging from neurotransmitter release and synaptic plasticity to gene expression and cell survival. In neurons, the maintenance of a steep transmembrane calcium gradient—typically maintained at concentrations of approximately 100 nM in the cytosol versus several millimolar in the extracellular space and endoplasmic reticulum—is fundamental to cellular viability. This gradient is actively preserved by a network of calcium pumps, exchangers, and buffers that tightly regulate the influx and efflux of calcium across the plasma membrane and intracellular organelles. Disruption of this gradient, whether through excitotoxicity, metabolic failure, oxidative stress, or physical membrane rupture, leads to a rapid and uncontrolled rise in cytosolic calcium concentration. This elevation triggers a cascade of destructive enzymatic activities, including the activation of calpains, phospholipases, and endonucleases, which collectively dismantle cellular architecture and initiate irreversible cell death. While many conventional assays for cell death rely on downstream consequences such as DNA fragmentation, phosphatidylserine externalization, or dye permeability, these markers often manifest after the point of no return has been crossed and may be delayed, inconsistent, or confounded by the heterogeneity of cell death pathways. Moreover, in live-cell imaging contexts, the persistence of fluorescent debris or the slow kinetics of dye uptake can obscure the precise temporal onset of death. The calcium ion, by contrast, acts as an immediate and universal signal of membrane failure, as its uncontrolled influx is among the earliest and most consistent events in the terminal phase of cell demise across diverse cell types and death modalities. Therefore, a biosensor capable of detecting the precise moment when calcium homeostasis collapses offers a uniquely sensitive, pathway-agnostic, and temporally precise means of defining cell death, enabling the accurate delineation of the transition from a dying to a dead cell in longitudinal studies.

## SUMMARY

- summarize methods for detecting cell death

Methods for detecting cell death involve the expression of genetically encoded polypeptide biosensors within living cells that respond to the loss of calcium homeostasis by producing a measurable fluorescent signal. These methods utilize engineered calcium-binding domains with modified affinity characteristics to ensure specificity for pathological, rather than physiological, calcium elevations. The detection of cell death is achieved by monitoring the increase in fluorescence intensity of the biosensor relative to a constitutively expressed reference fluorophore, allowing for pseudo-ratiometric quantification that corrects for variations in expression level and optical path length. The signal is acquired through time-lapse fluorescence microscopy, and a predefined threshold, empirically derived from control populations exposed to known cytotoxic agents, is applied to classify cells as alive or dead. This approach enables automated, high-throughput, single-cell resolution analysis of cell death dynamics across heterogeneous populations over extended periods, without the need for exogenous dyes or fixation.

- define isolated nucleic acid sequences

Isolated nucleic acid sequences refer to DNA or RNA molecules that have been removed from their natural cellular context and are substantially free of other cellular components, including genomic DNA, RNA, proteins, lipids, or metabolites, with which they are typically associated in vivo. These sequences are chemically or enzymatically purified and may be synthesized, cloned, or amplified using molecular biology techniques. In the context of this invention, isolated nucleic acid sequences encode the genetically encoded death indicator polypeptide, including the modified calcium-binding domain, the circularly permuted fluorescent protein, the self-cleaving P2a peptide, and the reference fluorophore, each operably linked under the control of a transcriptional regulatory element. These sequences are capable of being introduced into host cells, where they are transcribed and translated to produce the functional biosensor, and are distinct from naturally occurring nucleic acids in their engineered composition and intended function.

- describe vectors and cells

Vectors are nucleic acid constructs designed for the delivery and expression of genetic material within host cells. In this invention, vectors are engineered to contain the isolated nucleic acid sequence encoding the GEDI polypeptide, along with regulatory elements such as promoters, enhancers, and polyadenylation signals, to ensure robust and cell-type-specific expression. These vectors may be plasmid-based, viral, or synthetic and are capable of stably or transiently integrating into the host genome or remaining episomal. Cells used in these methods include primary neurons, induced pluripotent stem cell-derived neurons, neural cell lines, and other eukaryotic cell types that maintain calcium homeostasis and undergo regulated cell death. The cells are transfected or transduced with the vector to express the GEDI biosensor, and their viability is monitored over time using fluorescence imaging to detect the onset of death.

- summarize methods for monitoring calcium flux

Methods for monitoring calcium flux involve the use of genetically encoded biosensors that alter their fluorescence properties in response to changes in intracellular calcium concentration. In this invention, the GEDI biosensor is specifically engineered to detect supraphysiological calcium elevations associated with cell death, rather than transient calcium spikes linked to neuronal firing. Monitoring is performed by acquiring time-lapse fluorescence images of cells expressing the GEDI construct using appropriate excitation and emission filters, and the resulting signal is normalized to a co-expressed reference fluorophore to generate a ratiometric readout. The rate and magnitude of signal increase are analyzed to determine the temporal dynamics of calcium dysregulation, and the signal is compared to a predefined death threshold to classify cellular fate.

- summarize methods for monitoring cell death

Methods for monitoring cell death employ the GEDI biosensor to detect the irreversible loss of calcium homeostasis as a definitive marker of cellular demise. Cells expressing the biosensor are imaged longitudinally using fluorescence microscopy, and the pseudo-ratiometric signal is quantified over time. A death threshold is established based on the mean signal of live cells and the mean signal of cells exposed to cytotoxic agents, and any cell whose GEDI ratio exceeds this threshold is classified as dead. This method enables the automated, objective, and high-resolution tracking of individual cell death events in complex populations, without reliance on morphological changes or exogenous dyes, and is applicable to both in vitro and in vivo systems.

- summarize methods for screening agents

Methods for screening agents involve exposing cells expressing the GEDI biosensor to libraries of chemical compounds, biologics, or genetic perturbations and monitoring the rate and extent of GEDI signal increase as a readout of cytotoxicity or neuroprotection. Automated imaging systems capture time-lapse data across multi-well plates, and algorithmic analysis classifies each well based on the proportion of cells exceeding the death threshold. Agents that delay or prevent the rise in GEDI signal are identified as potential neuroprotective compounds, while those that accelerate the signal are flagged as cytotoxic. This platform enables high-throughput, quantitative, and physiologically relevant screening of therapeutic candidates for neurodegenerative diseases.

- specify tissue specific promoters

Tissue-specific promoters are DNA sequences that drive gene expression preferentially in defined cell types or tissues. In this invention, promoters such as the human synapsin 1 promoter, the neuroD promoter, the myosin heavy chain promoter, and the glial fibrillary acidic protein promoter are used to restrict expression of the GEDI biosensor to neurons, motor neurons, cardiac myocytes, or astrocytes, respectively. These promoters ensure that the biosensor is expressed only in the target cell population, minimizing off-target signals and enabling precise monitoring of cell death in complex tissues or in vivo models.

- specify fluorescent labels

Fluorescent labels used in this invention include red fluorescent proteins such as mRuby, mApple, and TagRFP, green fluorescent proteins such as EGFP and eGFP, blue fluorescent proteins such as TagBFP2 and mTagBFP2, and variants thereof, each selected for brightness, photostability, and spectral separation. These labels are fused to the GEDI construct either as the calcium-responsive reporter or as the constitutive reference fluorophore, enabling ratiometric detection. The fluorescent labels are chosen to minimize spectral overlap with endogenous autofluorescence and to permit multiplexing with other biosensors.

- specify distinguishable markers

Distinguishable markers refer to fluorescent proteins or other detectable molecules that can be simultaneously imaged alongside the GEDI biosensor without spectral interference. These include fluorescent proteins emitting in distinct wavelength ranges such as cyan, yellow, or far-red, as well as non-fluorescent markers such as luciferase or enzymatic reporters. These markers enable the concurrent monitoring of calcium flux, apoptosis, autophagy, mitochondrial membrane potential, or gene expression in the same cell, allowing for the dissection of complex cellular responses to stress or therapeutic intervention.

- specify modified calcium binding motifs

Modified calcium binding motifs are engineered variants of naturally occurring calcium-binding domains, such as calmodulin, troponin C, or EF-hand motifs, in which specific amino acid residues have been substituted to alter calcium affinity, kinetics, or cooperativity. In this invention, these motifs are modified to reduce calcium binding affinity such that the dissociation constant (Kd) is shifted from the nanomolar to the micromolar range, rendering the sensor unresponsive to physiological calcium transients but highly sensitive to the millimolar calcium concentrations that occur upon plasma membrane rupture. These modifications are achieved through site-directed mutagenesis, domain swapping, or directed evolution.

- specify cell types

Cell types to which the methods of this invention are applicable include primary cortical neurons, hippocampal neurons, motor neurons derived from induced pluripotent stem cells, cerebellar granule cells, cardiomyocytes, astrocytes, microglia, HEK293 cells, and other mammalian and non-mammalian cell lines that maintain calcium homeostasis and undergo regulated cell death. The invention is further applicable to neurons within organotypic brain slice cultures and to neurons in live zebrafish larvae, enabling the study of cell death in both simplified and physiologically relevant tissue contexts.

## DETAILED DESCRIPTION

- introduce purpose of detailed description

The purpose of this detailed description is to provide a comprehensive and enabling disclosure of the invention, including the composition, construction, and application of genetically encoded death indicators, as well as the methods for their use in detecting, quantifying, and analyzing cell death across diverse biological systems. This section elaborates on the structural components of the biosensor, the vectors and expression systems used for delivery, the cellular and animal models in which the biosensor is functional, and the analytical techniques that enable its reliable and reproducible implementation in both research and screening contexts.

- explain organization of detailed description

The detailed description is organized to first define key terms and concepts essential to understanding the invention, followed by a comprehensive exposition of the vectors and expression systems used to deliver the GEDI biosensor, the cell types and animal models in which it functions, the methods for its application in monitoring calcium flux and cell death, and the kits and reagents that facilitate its use. Each subsection provides sufficient detail to enable a person of ordinary skill in the art to make and use the invention without undue experimentation.

- discuss conventional techniques used in practice

Conventional techniques used in the practice of this invention include molecular cloning, transfection and transduction of mammalian cells, fluorescence microscopy, time-lapse imaging, image segmentation and tracking, ratiometric fluorescence analysis, and statistical modeling of survival curves. These techniques are well established in the fields of cell biology, neuroscience, and high-throughput screening and are employed here in novel combinations to detect cell death with unprecedented temporal precision and objectivity.

- provide references for conventional techniques

References for conventional techniques include standard texts such as Molecular Cloning: A Laboratory Manual by Sambrook and Russell, Current Protocols in Neuroscience, and Methods in Enzymology, as well as peer-reviewed publications detailing the use of GCaMP, P2a peptides, confocal microscopy, and automated image analysis in neuronal cultures and zebrafish models.

- explain use of numerical designations

Numerical designations such as “GEDI-P2a-EGFP” or “GC150-NLS-P2a-mApple-NLS” are used herein to denote specific embodiments of the biosensor, where the first component indicates the calcium-sensing domain, the second indicates the self-cleaving peptide, and the third indicates the reference fluorophore, with optional additions such as nuclear localization signals denoted by “NLS.” These designations allow for precise identification and reproducible replication of each construct.

- discuss interpretation of singular and plural forms

As used herein, the singular forms “a,” “an,” and “the” include plural referents unless the context clearly dictates otherwise. Thus, a “vector” may refer to a single construct or to a collection of constructs, and a “cell” may refer to one or more cells within a population, depending on the context of the claim or description.

- introduce definitions of terms used

The following terms are used throughout this disclosure with the meanings ascribed herein to ensure clarity and consistency. These definitions are intended to be binding and shall govern the interpretation of all claims and descriptions.

- define "about" and its usage

The term “about” as used herein refers to a range of values that encompasses the stated value plus or minus ten percent, unless otherwise indicated. For example, a Kd of about 10 μM encompasses values from 9 μM to 11 μM. This term is used to account for experimental variability, measurement error, and biological heterogeneity inherent in biological systems.

- define "comprising" and its usage

The term “comprising” is used in its open-ended sense, meaning that the recited elements or steps are included but are not exclusive. A composition comprising a GEDI polypeptide may contain additional components such as other biosensors, dyes, or media components, and a method comprising imaging cells may include additional steps such as fixation, staining, or data normalization.

### Definitions

- define "about"

As defined above, “about” denotes a range of ±10% around a stated numerical value, accommodating natural variability in biological measurements and experimental conditions.

- define "acceptable", "effective", or "sufficient"

The terms “acceptable,” “effective,” and “sufficient” are used interchangeably to describe a quantity, concentration, or condition that achieves the intended biological outcome without causing undue toxicity, artifact, or interference. For example, an effective concentration of a cytotoxic agent is one that induces cell death in a reproducible manner without inducing non-specific stress responses.

- define "and/or"

The term “and/or” means that one, more than one, or all of the listed elements may be present or performed. For example, a vector comprising a promoter and/or a polyadenylation signal may contain either one or both elements.

- define "comprising"

As defined above, “comprising” is an open-ended term that permits the inclusion of additional elements not explicitly recited.

- define "consisting essentially of"

The phrase “consisting essentially of” limits the scope of the claim to the specified elements and those that do not materially affect the basic and novel characteristics of the invention. For example, a vector consisting essentially of a GEDI polypeptide coding sequence and a promoter may include additional elements such as a selectable marker if those elements do not alter the biosensor’s ability to detect cell death.

- define "consisting of"

The phrase “consisting of” excludes any element, step, or ingredient not specified. A composition consisting of a GEDI polypeptide and a reference fluorophore contains no other components.

- define "isolated"

An “isolated” nucleic acid or polypeptide is one that has been separated from its natural cellular environment and is substantially free of other cellular components with which it is normally associated in vivo.

- define "modified"

“Modified” refers to a nucleic acid or polypeptide that has been altered from its naturally occurring form through chemical, enzymatic, or genetic means, including point mutations, insertions, deletions, or fusion to heterologous sequences.

- define "nucleic acid"

A “nucleic acid” refers to a polymer of deoxyribonucleotides or ribonucleotides, including DNA, RNA, cDNA, synthetic analogs, and modified backbones, capable of hybridization or encoding a polypeptide.

- define "recombinant nucleic acid"

A “recombinant nucleic acid” is a nucleic acid molecule that has been artificially constructed by joining nucleic acid segments from different sources, such as a GEDI coding sequence fused to a neuronal promoter.

- define "operably linked"

“Operably linked” describes a functional relationship between two or more genetic elements, such that one element regulates the expression or function of the other. For example, a promoter is operably linked to a GEDI coding sequence when it drives transcription of that sequence.

- define "protein"

A “protein” refers to a polypeptide chain, whether naturally occurring or synthetic, that may be post-translationally modified and may include non-peptidyl moieties such as fluorescent tags or affinity handles.

- define "sequence identity"

“Sequence identity” refers to the percentage of identical amino acid or nucleotide residues between two aligned sequences over a specified length, determined using standard alignment algorithms such as BLAST or Needleman-Wunsch.

- define "variant"

A “variant” is a nucleic acid or polypeptide that differs from a reference sequence by one or more substitutions, insertions, or deletions, yet retains the essential functional properties of the original, such as the ability to detect cell death.

- define "vector"

A “vector” is a nucleic acid construct capable of delivering and expressing a transgene in a host cell, including plasmids, viral particles, and synthetic delivery systems.

- provide additional definitions

Additional definitions include “self-cleaving peptide,” which refers to a short amino acid sequence such as P2a that mediates co-translational cleavage between two fused polypeptides; “pseudo-ratiometric,” which refers to the normalization of a signal to a co-expressed reference fluorophore without requiring simultaneous excitation at two wavelengths; and “time-lapse imaging,” which refers to the acquisition of images at regular intervals over an extended period to monitor dynamic biological processes.

### Vectors

- introduce vectors

Vectors are essential tools for the delivery and expression of the GEDI biosensor in target cells. These constructs are designed to ensure high-level, stable, and cell-type-specific expression of the biosensor while minimizing cytotoxicity and silencing.

- describe expression vectors

Expression vectors contain all necessary elements for transcription and translation of the GEDI polypeptide, including a promoter, a coding sequence, a polyadenylation signal, and optionally, a selectable marker and origin of replication.

- list examples of mammalian expression vectors

Examples of mammalian expression vectors include pCMV, pCAG, pEF, pLenti, and pAAV, each of which has been adapted to carry the GEDI cassette under the control of a neuron-specific promoter.

- describe viral vectors

Viral vectors are engineered from naturally occurring viruses to deliver genetic material into cells with high efficiency. These include lentiviruses, retroviruses, adeno-associated viruses, and adenoviruses.

- list examples of viral vectors

Examples of viral vectors include pLenti-CMV, pRRLsin.PPT.hSyn1, pAAV-hSyn1, and pAdeno.hSyn1, each of which has been modified to contain the GEDI expression cassette.

- describe retrovirus-based vectors

Retrovirus-based vectors integrate into the host genome, enabling long-term expression of the biosensor in dividing cells. These vectors are derived from murine leukemia virus and are commonly used for stable transduction of primary neurons.

- describe lentivirus-based vectors

Lentivirus-based vectors are derived from HIV-1 and are capable of transducing both dividing and non-dividing cells, making them ideal for delivery into post-mitotic neurons. These vectors feature self-inactivating (SIN) designs to reduce the risk of insertional mutagenesis.

- describe recombinant retrovirus production

Recombinant retroviruses are produced by transfecting packaging cell lines with a transfer vector encoding the GEDI cassette and separate plasmids encoding the viral structural proteins, such as gag, pol, and env.

- describe packaging cells

Packaging cells are specialized cell lines, such as HEK293T or GP2-293, that express the viral structural proteins required for virion assembly but lack the packaging signal, thereby preventing self-replication.

- list examples of packaging cell lines

Examples of packaging cell lines include HEK293T, GP2-293, and Phoenix-Eco, each of which is used to produce high-titer retroviral or lentiviral particles.

- describe retroviral construct components

Retroviral constructs contain a 5′ long terminal repeat (LTR), a packaging signal (ψ), the GEDI expression cassette, and a 3′ LTR, with optional internal promoters and WPRE elements to enhance expression.

- list examples of retroviral constructs

Examples of retroviral constructs include pMSCV-GEDI-P2a-EGFP, pMIG-GEDI-P2a-3xBFP, and pLZRS-GEDI-NLS-P2a-EGFP-NLS.

- describe self-inactivating lentiviral vectors

Self-inactivating lentiviral vectors contain deletions in the 3′ LTR that are copied to the 5′ LTR during reverse transcription, rendering the integrated provirus transcriptionally inactive and reducing the risk of oncogene activation.

- describe virus vector plasmids

Virus vector plasmids are plasmid constructs that contain the full viral genome with the GEDI cassette inserted in place of viral genes, and are used to produce viral particles upon transfection into packaging cells.

- list examples of virus vector plasmids

Examples include pLenti-hSyn1-GEDI-P2a-EGFP, pAAV-hSyn1-GC150-P2a-mApple, and pLenti-neuroD-GEDI-P2a-EGFP.

- describe methods of producing recombinant viruses

Recombinant viruses are produced by transfecting packaging cells with the virus vector plasmid and helper plasmids encoding viral structural proteins, followed by collection of viral supernatant 48–72 hours post-transfection.

- describe introduction of viral construct into packaging cell

The viral construct is introduced into packaging cells via calcium phosphate precipitation, lipofection, electroporation, or nucleofection, using methods optimized for high transfection efficiency.

- list transfection methods

Transfection methods include lipid-based reagents such as Lipofectamine 2000 and 3000, electroporation using the Neon system, nucleofection with Amaxa kits, and calcium phosphate precipitation.

- describe non-viral based transfection methods

Non-viral methods include plasmid DNA delivery via microinjection, particle bombardment, or polymer-based nanoparticles, which are particularly useful for in vivo applications where viral vectors are contraindicated.

- describe expression control sequences

Expression control sequences include promoters, enhancers, introns, and untranslated regions that regulate the timing, level, and cell-type specificity of GEDI expression.

- describe promoters

Promoters are DNA sequences upstream of the coding region that recruit RNA polymerase and transcription factors to initiate transcription. In this invention, promoters are selected for their ability to drive strong, sustained expression in specific cell types.

- list examples of eukaryotic promoters

Examples include the cytomegalovirus immediate early promoter, the elongation factor 1 alpha promoter, the chicken β-actin promoter, and the human synapsin 1 promoter.

- describe tissue-specific promoters

Tissue-specific promoters restrict expression to particular cell types. Examples include the neuroD promoter for neurons, the myosin light chain promoter for cardiomyocytes, and the GFAP promoter for astrocytes.

- describe enhancers

Enhancers are distal regulatory elements that increase transcriptional activity of a promoter. In this invention, enhancers such as the CMV enhancer or the WPRE element are used to boost GEDI expression levels.

- describe detectable markers

Detectable markers are genes or proteins that produce a measurable signal, such as fluorescence, luminescence, or enzymatic activity, and are used to confirm successful transfection or transduction.

- introduce cell death detection

Cell death detection is achieved through the GEDI biosensor, which converts the loss of calcium homeostasis into a fluorescent signal that is quantifiable, reproducible, and independent of morphological changes.

- importance of calcium gradient

The maintenance of the calcium gradient across the plasma membrane is essential for cellular viability, and its collapse is a universal and early event in cell death, making it an ideal target for biosensor design.

- describe fluorescent detectable signals

Fluorescent detectable signals are generated by the conformational change of the circularly permuted fluorescent protein upon calcium binding, resulting in increased quantum yield and emission intensity.

- introduce isolated nucleic acid sequence

The isolated nucleic acid sequence encoding the GEDI polypeptide is synthesized de novo or assembled from modular parts and is free of endogenous regulatory elements that might interfere with expression.

- describe detectable markers

Detectable markers include fluorescent proteins, luciferases, and enzymes such as β-galactosidase, which are used to verify expression and localization of the GEDI biosensor.

- introduce luminescence or fluorescence assays

Luminescence or fluorescence assays are used to quantify the GEDI signal in real time, using plate readers, microscopes, or flow cytometers equipped with appropriate filters.

- describe pseudo-ratiometric standardization

Pseudo-ratiometric standardization involves calculating the ratio of the GEDI signal to the constitutive reference fluorophore, thereby correcting for variations in expression level, cell thickness, and photobleaching.

- introduce examples of fluorescent detectable markers

Examples include EGFP, mApple, TagBFP2, and mCherry, each of which is selected for spectral compatibility and brightness.

- describe RGEDI-P2A-X embodiments

RGEDI-P2A-X embodiments refer to constructs in which the red GEDI sensor is fused to a P2a peptide and a reference fluorophore X, such as EGFP, 3xBFP, or mApple, enabling multiplexed imaging.

- introduce proteins conferring resistance

Proteins conferring resistance, such as neomycin phosphotransferase or puromycin N-acetyltransferase, are optionally included in the vector to allow selection of stably transfected cells.

- introduce calcium binding motif

The calcium binding motif is derived from calmodulin or troponin C and has been modified to exhibit a Kd in the micromolar range, ensuring specificity for pathological calcium elevations.

- describe GCaMP variants

GCaMP variants are calcium indicators derived from circularly permuted GFP fused to calmodulin and M13, and are used as templates for engineering GEDI sensors with altered affinity.

- introduce modified calcium binding motif

The modified calcium binding motif contains amino acid substitutions in the EF-hand loops that reduce calcium affinity, such as D130A, E134A, or D135A, to shift the Kd from 200 nM to 5–10 μM.

- describe nucleotide sequence identity

Nucleotide sequence identity refers to the percentage of identical bases between the GEDI coding sequence and a reference sequence, with preferred embodiments exhibiting at least 85% identity.

- introduce polypeptide sequence identity

Polypeptide sequence identity refers to the percentage of identical amino acids between the GEDI polypeptide and a reference sequence, with preferred embodiments exhibiting at least 90% identity.

- describe EF-hand motifs

EF-hand motifs are helix-loop-helix structural domains that bind calcium ions, and are the basis for the calcium-sensing function of calmodulin and troponin C.

- introduce Troponin C

Troponin C is a calcium-binding protein from muscle tissue that contains multiple EF-hand motifs and has been used as a scaffold for engineering calcium sensors with tailored affinity.

- describe optimized signal

Optimized signal refers to the maximal signal-to-noise ratio achieved by tuning the calcium affinity, fluorophore brightness, and expression level to ensure robust detection of death without false positives.

- introduce reduced Ca2+ binding affinity

Reduced Ca2+ binding affinity is achieved by mutating key calcium-coordinating residues in the EF-hand motifs, resulting in a biosensor that responds only to calcium concentrations exceeding 5 μM.

- describe modified calmodulin

Modified calmodulin refers to calmodulin with amino acid substitutions in the calcium-binding loops that reduce its affinity for calcium while preserving structural integrity.

- introduce dissociation constant

The dissociation constant (Kd) is the concentration of calcium at which half of the binding sites are occupied, and is a key parameter defining the sensitivity of the GEDI biosensor.

- describe localization sequence

Localization sequences, such as nuclear localization signals (NLS) or mitochondrial targeting sequences, direct the GEDI biosensor to specific subcellular compartments to enhance signal detection or to monitor compartment-specific calcium dysregulation.

- introduce GEDI constructs

GEDI constructs are the complete nucleic acid or polypeptide assemblies that encode the functional biosensor, including the calcium-binding domain, the fluorescent reporter, the P2a peptide, and the reference fluorophore, all operably linked under a tissue-specific promoter.

### Cells

- define cells

Cells are the basic structural and functional units of living organisms, and in the context of this invention, refer to eukaryotic cells capable of maintaining calcium homeostasis and undergoing regulated cell death.

- specify cell types

Cell types include primary neurons, induced pluripotent stem cell-derived neurons, motor neurons, astrocytes, microglia, cardiomyocytes, HEK293 cells, and other cell lines that exhibit calcium-dependent viability.

- describe neural cells

Neural cells include neurons, astrocytes, oligodendrocytes, and microglia, and are the primary target of the GEDI biosensor due to their high susceptibility to calcium dysregulation in neurodegenerative conditions.

- describe cardiac cells

Cardiac cells, including cardiomyocytes and pacemaker cells, are susceptible to calcium overload during ischemia and reperfusion injury, and are suitable for adaptation of the GEDI biosensor to cardiovascular disease models.

### Animals

- define animals

Animals are multicellular eukaryotic organisms used as model systems for in vivo studies of cell death, including rodents, zebrafish, and other vertebrates.

- specify animal types

Animal types include mice, rats, zebrafish, and non-human primates, each of which may be used to evaluate the GEDI biosensor in physiological and pathological contexts.

- describe transgenic animals

Transgenic animals are genetically modified to express the GEDI biosensor under the control of a tissue-specific promoter, enabling longitudinal imaging of cell death in intact tissues.

- explain method for making transgenic animals

Transgenic animals are made by microinjection of the GEDI expression vector into fertilized eggs, followed by implantation into surrogate mothers and screening of offspring for transgene expression.

- describe embryonic stem cells

Embryonic stem cells are pluripotent cells derived from the inner cell mass of blastocysts and are used to generate chimeric animals or to differentiate into neurons for in vitro studies.

- describe induced pluripotent stem cells

Induced pluripotent stem cells are somatic cells reprogrammed to a pluripotent state using transcription factors such as Oct4, Sox2, Klf4, and c-Myc, and are used to generate patient-specific neurons for disease modeling.

- explain organotypic slice cultures

Organotypic slice cultures are thin sections of brain tissue maintained in vitro that preserve the three-dimensional architecture and cellular interactions of the intact brain, and are used to study cell death in a more physiologically relevant context.

- describe brain slice cultures

Brain slice cultures are prepared from embryonic or postnatal brain tissue and maintained on membrane inserts in serum-free medium, allowing for long-term imaging of neuronal death using the GEDI biosensor.

- specify regions of brain for organotypic slice cultures

Regions include the hippocampus, cortex, cerebellum, and spinal cord, each of which is relevant to different neurodegenerative diseases.

- describe use of organotypic slice cultures

Organotypic slice cultures are used to model neurodegenerative disease progression, test neuroprotective agents, and study the spread of cell death in a tissue context that retains synaptic connectivity.

### Methods of Use

- describe method for monitoring calcium flux

The method for monitoring calcium flux involves transducing cells with the GEDI biosensor, acquiring time-lapse fluorescence images using a confocal or spinning disk microscope, and calculating the ratiometric signal over time to detect pathological calcium elevations.

- describe method for monitoring cell death

The method for monitoring cell death involves establishing a death threshold based on control populations, applying this threshold to longitudinal imaging data, and classifying each cell as alive or dead based on whether its GEDI ratio exceeds the threshold.

- describe method for screening for agents

The method for screening for agents involves exposing multi-well plates of GEDI-expressing cells to compound libraries, acquiring automated time-lapse images, and using algorithmic analysis to identify compounds that delay or accelerate the GEDI signal increase.

- specify optical means for methods

Optical means include widefield, confocal, and spinning disk microscopy systems equipped with appropriate filters, laser sources, and high-sensitivity cameras for fluorescence detection.

- describe use of microscopes

Microscopes are used to capture high-resolution, time-lapse images of GEDI-expressing cells in culture or in vivo, with objectives optimized for depth penetration and photostability.

- describe automated time-lapse confocal microscopy

Automated time-lapse confocal microscopy involves the use of motorized stages, environmental control chambers, and software to repeatedly image the same field of view over days or weeks, enabling long-term tracking of individual cells.

- describe high-throughput and automated methods

High-throughput and automated methods involve the use of 96- or 384-well plates, robotic liquid handlers, and machine learning algorithms to process thousands of cells per experiment with minimal human intervention.

- specify signal intensity ratios for cell death

Signal intensity ratios for cell death are defined as the ratio of the GEDI signal to the reference fluorophore signal, with a threshold set at the mean of live cells plus 2.5 standard deviations.

- describe empirical determination of signal intensity ratios

Empirical determination involves exposing a subset of cells to known cytotoxic agents, measuring the resulting GEDI ratio, and using statistical analysis to define a threshold that distinguishes live from dead cells with high specificity.

### Kits

- describe kits and their components

Kits comprise one or more vectors encoding the GEDI biosensor, transfection reagents, positive and negative control plasmids, instructions for use, and optionally, a pre-calibrated death threshold value. Kits may be supplied as lyophilized or frozen reagents and are designed for immediate use in research or screening laboratories.

## EXAMPLES

### Example 1

- introduce genetically encoded calcium indicators (GECIs)

Genetically encoded calcium indicators (GECIs) are fluorescent biosensors derived from the fusion of circularly permuted fluorescent proteins with calcium-binding domains such as calmodulin and M13. These indicators have been widely used to monitor neuronal activity through the detection of transient increases in cytosolic calcium.

- describe GECI use in detecting Ca2+ transients

GECIs such as GCaMP6f exhibit a rapid increase in fluorescence in response to physiological calcium transients associated with action potentials and synaptic activity, making them indispensable tools for functional imaging in neuroscience.

- discover GECI use in detecting cell death

It was discovered that under conditions of cell death, GECIs such as GCaMP6f also exhibit a sustained increase in fluorescence due to the uncontrolled influx of extracellular calcium following plasma membrane rupture, suggesting that these sensors could be repurposed as death indicators.

- describe experimental setup for rat primary cortical neuron cultures

Rat primary cortical neurons were cultured from embryonic day 18–19 pups, transfected with plasmids encoding GCaMP6f or RGEDI-P2a-EGFP, and maintained in Neurobasal medium supplemented with B27 and GlutaMAX.

- analyze GCaMP6f expression in dying neurons

In neurons undergoing death due to NaN3 exposure, GCaMP6f exhibited a sustained increase in fluorescence that was indistinguishable from its response to physiological stimulation, making it impossible to differentiate between activity and death.

- show specificity of RGEDI response to cell death

In contrast, RGEDI, engineered with a modified calcium-binding domain, showed no response to electrical stimulation but exhibited a robust and sustained increase in fluorescence only upon exposure to NaN3, demonstrating its specificity for cell death.

- demonstrate multiplexing of RGEDI with other biosensors

RGEDI was successfully co-expressed with other biosensors such as Caspase3/7 reporters and mitochondrial membrane potential indicators, enabling simultaneous monitoring of multiple death pathways.

- analyze RGEDI expression in dying and healthy neurons

Live neurons expressing RGEDI-P2a-EGFP exhibited low red fluorescence and bright green fluorescence, whereas dying neurons showed a shift to yellow due to the increase in red signal, providing a clear visual distinction.

- show decay rates of RGEDI and EGFP fluorescence

Following cell death, the decay rates of RGEDI and EGFP fluorescence were found to be nearly identical, indicating that the GEDI ratio remains stable over time and is not subject to differential degradation.

- summarize advantages of GEDIs over other cell death indicators

GEDIs provide earlier, more specific, and more quantitative detection of cell death than traditional dyes or morphological criteria, and are non-toxic, enabling long-term imaging without perturbing the system.

- conclude GEDIs as alternative cell death indicators

GEDIs represent a novel class of cell death indicators that provide an unambiguous, pathway-independent, and temporally precise marker of neuronal demise, offering a superior alternative to existing methods.

### Example 2

- describe limitations of GCaMP6f-P2A-mRuby

GCaMP6f-P2a-mRuby, while useful for detecting calcium transients, lacks specificity for cell death because its signal increases in response to both physiological activity and pathological calcium influx, leading to false positives.

- design novel genetically encoded death indicators (GEDIs)

To overcome this limitation, a novel GEDI was designed by modifying the calcium-binding domain of CEPIA to reduce its affinity for calcium, resulting in a sensor that responds only to the millimolar calcium concentrations that occur upon membrane rupture.

- characterize GEDI variants

Multiple GEDI variants were generated, including RGEDI-P2a-EGFP, RGEDI-P2a-3xBFP, GC150-P2a-mApple, and nuclear-localized versions, each exhibiting distinct spectral properties and subcellular targeting.

- show predicted Kd values for Ca2+ binding

Computational modeling and in vitro binding assays predicted that RGEDI has a Kd of approximately 8 μM, while GC150 has a Kd of approximately 3 μM, confirming their suitability for detecting pathological calcium elevations.

### Example 3

- describe use of GEDIs as in vitro cell death reporters

GEDIs were used as in vitro reporters of neuronal death in primary rat cortical neuron cultures exposed to a variety of cytotoxic agents, including glutamate, NaN3, and disease-associated proteins.

- express RGEDI-P2A-EGFP in rat cortical primary neurons

Neurons were transfected with hSyn1:RGEDI-P2a-EGFP and imaged every 3 hours over 72 hours using automated confocal microscopy.

- image and track individual neurons over time

Individual neurons were tracked across multiple time points using custom segmentation and tracking algorithms, allowing for the longitudinal analysis of GEDI signal dynamics.

- establish "live" and "dead" GEDI signal ratios

A GEDI threshold was established by comparing the signal ratios of live neurons to those of neurons treated with NaN3, defining a clear boundary between viable and non-viable cells.

- compute GEDI threshold

The GEDI threshold was computed using the equation: (mean dead ratio – mean live ratio) × 0.25 + mean live ratio, yielding a value of 0.05 for RGEDI-P2a-EGFP.

- classify cells as live or dead based on GEDI signal

Cells with a GEDI ratio above 0.05 were classified as dead, and those below as live, with 99.7% accuracy compared to manual scoring.

- show representative images of individual neurons

Representative images showed clear transitions from green/yellow (live) to yellow/red (dead) in neurons undergoing death, with no overlap between the two populations.

- plot RGEDI intensity over time

Time plots of GEDI intensity showed a sharp, step-like increase at the time of death, with no fluctuations prior to the threshold, confirming the irreversibility of the signal.

- analyze RGEDI signal in response to KCl exposure

Exposure to high KCl induced membrane depolarization but did not trigger a GEDI signal increase, confirming that the sensor does not respond to physiological calcium influx.

- show RGEDI signal in response to HttEx1-Q97 expression

Expression of mutant huntingtin induced a progressive increase in GEDI ratio, with death occurring over 48–72 hours, consistent with the slow progression of Huntington’s disease pathology.

- analyze RGEDI signal in response to NaN3 exposure

NaN3 induced a rapid and uniform increase in GEDI ratio within 5 minutes, confirming the sensor’s sensitivity to acute metabolic failure.

- show non-linear regression of RGEDI signal increase

Non-linear regression of the signal increase revealed a one-phase association kinetics profile, indicating a single, rapid transition from low to high signal upon death.

- analyze RGEDI signal in response to different neurodegenerative diseases

GEDI signal was elevated in neurons expressing α-synuclein, TDP43, and SOD1-D90A, demonstrating its utility across multiple neurodegenerative disease models.

- conclude GEDIs as reliable cell death reporters

GEDIs provide a highly reliable, quantitative, and automated method for detecting neuronal death across diverse experimental conditions and disease models.

### Example 4

- describe use of GEDIs as in vivo and in tissue cell death reporters

GEDIs were adapted for use in vivo in zebrafish larvae and in organotypic brain slice cultures to monitor neuronal death in intact tissue.

- co-inject zebrafish embryos with mnx1:Gal4, UAS:GCaMP6f, and UAS:Nit

Zebrafish embryos were co-injected with plasmids encoding mnx1:Gal4, UAS:GCaMP6f, and UAS:NTR, and treated with metronidazole to induce motor neuron ablation.

- image motor neurons in zebrafish larvae

Motor neurons were imaged using confocal microscopy over 48 hours, and GCaMP6f signal increased in response to NTR-mediated ablation, but also during normal neuronal activity.

- analyze GCaMP signal in response to MTZ treatment

GCaMP6f signal increased during both NTR-mediated death and spontaneous neuronal activity, making it unsuitable as a specific death indicator in vivo.

- co-transfect hippocampal slice culture with RGEDI-P2a-EGFP and htt ex1 97Q

Hippocampal slices were transfected with RGEDI-P2a-EGFP and mutant huntingtin, and death was monitored over 72 hours using automated imaging.

- conclude GEDIs as useful in vivo and in tissue

GEDIs, particularly GC150-P2a-mApple, provided a specific and sensitive readout of neuronal death in vivo, without requiring immobilization or pharmacological suppression of activity, establishing their utility in complex tissue environments.

### Example 6

- describe use of GEDIs with robotic microscopy

GEDIs were integrated with robotic microscopy platforms to enable fully automated, high-throughput analysis of neuronal survival across thousands of cells.

- classify neurons as dead or alive using GEDI signal

Using the empirically derived GEDI threshold, neurons were automatically classified as alive or dead without human intervention.

- compare survival analysis using automated and manual analysis

Automated analysis using GEDI thresholds showed 99.7% concordance with manual scoring, while manual scoring alone exhibited significant inter-rater variability and missed early death events.

- show hazard ratios and statistical analysis

Kaplan-Meier survival curves and Cox proportional hazards models generated from GEDI data revealed statistically significant differences in survival between disease models and controls.

- describe advantages of GEDI automated analysis

Automated analysis using GEDI eliminated observer bias, enabled real-time data processing, and allowed for the detection of subtle survival differences that were invisible to manual scoring.

- show misclassification of manually curated data

Manual scoring misclassified 12% of neurons as alive when they had already exceeded the GEDI death threshold, and these neurons were confirmed dead at the next time point.

- describe use of GEDIs in drug screening

GEDIs were used to screen a library of 1,200 compounds for neuroprotective activity, identifying several compounds that delayed the GEDI signal increase in neurons exposed to glutamate.

- expose neurons to glutamate and analyze response

Neurons expressing GEDI were exposed to 0.1 mM glutamate, and the time to death was recorded for each cell, revealing a bimodal distribution of sensitivity.

- conclude GEDIs as useful for screening and identifying physiological characteristics

GEDIs enable the identification of neuroprotective agents, the characterization of neuronal subpopulations with differential vulnerability, and the quantification of disease progression in a manner that is scalable, objective, and physiologically relevant.