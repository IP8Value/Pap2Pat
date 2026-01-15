# DESCRIPTION

## STATEMENT AS TO RIGHTS TO INVENTIONS MADE UNDER FEDERALLY SPONSORED RESEARCH AND DEVELOPMENT

This invention was made with government support under Grant No. [REDACTED] awarded by the National Institutes of Health. The government has certain rights in the invention.

## BACKGROUND OF THE INVENTION

The origins of primordial germ cells (PGCs) in mammalian embryogenesis have long been a subject of scientific inquiry due to their essential role in transmitting genetic and epigenetic information across generations. In model organisms such as mice, PGCs arise from the posterior epiblast during early post-implantation development, coinciding with the formation of the primitive streak—a transient structure that gives rise to mesoderm and endoderm. However, in humans, the precise developmental window for PGC specification occurs between weeks 2 and 3 of embryogenesis, a period that is ethically and technically inaccessible for direct experimental analysis. Consequently, much of our understanding of human PGC development has been inferred from comparative studies in non-human primates and rodents, as well as from in vitro differentiation of human pluripotent stem cells (hPSCs).

Previous studies have attempted to recapitulate human PGC specification using three-dimensional (3D) aggregate cultures of hPSCs, which are first exposed to primitive streak-inducing signals such as WNT and TGFβ agonists, followed by treatment with bone morphogenetic protein (BMP), stem cell factor (SCF), epidermal growth factor (EGF), and leukemia inhibitory factor (LIF) to promote germline commitment. While these approaches have yielded PGC-like cells (PGCLCs), they suffer from significant limitations. The 3D architecture introduces heterogeneity due to variable diffusion gradients of signaling molecules, inconsistent cell-cell interactions, and difficulties in precisely controlling temporal exposure to extracellular cues. Moreover, the resulting PGCLC populations are often contaminated with off-target lineages such as mesoderm or endoderm, and purification strategies have relied on surface markers that are also expressed on undifferentiated hPSCs, thereby limiting specificity and yield.

Controversies persist regarding the exact lineage relationship between human PGCs and somatic lineages. One prevailing model posits that PGCs and primitive streak derivatives share a common posterior epiblast precursor, with fate decisions governed by mutually exclusive signaling inputs. An alternative hypothesis suggests that PGCs arise independently from a distinct epiblast domain, possibly the dorsal amnion, as observed in cynomolgus monkey embryos. These unresolved questions underscore the need for a robust, reproducible, and scalable in vitro platform that enables precise dissection of the molecular and cellular events governing human PGC specification.

To address these challenges, the present invention provides a simplified two-dimensional (2D) monolayer culture system that overcomes the limitations of prior art by enabling temporally controlled manipulation of key signaling pathways—specifically WNT and TGFβ—followed by defined inhibition and activation steps that drive efficient and specific PGC formation. This platform not to only enhances the purity and reproducibility of PGCLC generation across diverse hPSC lines but also facilitates high-resolution molecular characterization and functional interrogation of human germline development.

## BRIEF SUMMARY OF THE INVENTION

The present invention provides a method for forming primordial germ cells (PGCs) from human pluripotent stem cells (hPSCs) in a two-dimensional monolayer culture system through temporally dynamic modulation of WNT and TGFβ signaling pathways. In one aspect, the method comprises contacting a population of hPSCs with a WNT agonist and a TGFβ agonist for a defined duration—preferably 12 hours—to induce a posterior epiblast intermediate state characterized by co-expression of pluripotency factors and early primitive streak markers. Subsequently, the WNT agonist and TGFβ agonist are removed, and the posterior epiblast cell population is contacted with a WNT inhibitor to suppress somatic differentiation and promote germline commitment.

The invention further provides a method for isolating PGCs based on a novel cell-surface marker profile: CXCR4-positive, PDGFRα-negative, and GARP-negative (CXCR4⁺PDGFRα⁻GARP⁻). This signature enables highly specific fluorescence-activated cell sorting (FACS) purification of PGCs from heterogeneous differentiation cultures without reliance on transgenic reporters. The isolated PGCs express canonical germline markers including NANOS3, PRDM1 (BLIMP1), TFAP2C, SOX17, OCT4, and NANOG, while lacking expression of endodermal (e.g., FOXA2, CDX2) or mesodermal (e.g., HAND1, ACTC1) lineage markers.

In another aspect, the invention outlines a complete protocol for forming PGCs within 3.5 days of differentiation, achieving average yields of 36.7% pure PGCs across multiple human embryonic stem cell (hESC) and induced pluripotent stem cell (hiPSC) lines. The method is serum-free, chemically defined, and scalable, making it suitable for drug screening, disease modeling, and infertility research.

Additionally, the invention describes a method for treating infertility by generating patient-specific PGCs from hiPSCs derived from individuals with genetic or idiopathic infertility, followed by potential maturation into functional gametes—a process that may eventually enable autologous reproductive therapies.

Various aspects of the resulting PGCs are detailed, including their transcriptional identity, which closely resembles early human fetal PGCs (FGC1 stage) as confirmed by single-cell RNA sequencing; their continuous expression of pluripotency factors such as NANOG throughout the transition from pluripotency to germline fate; and their functional responsiveness to germline-specifying extracellular cues including BMP4, SCF, and EGF at optimized concentrations significantly lower than those used in prior 3D protocols.

## DETAILED DESCRIPTION OF THE INVENTION

### Definitions

The present patent application relates to compositions and methods for the in vitro generation and isolation of human primordial germ cells (PGCs) from pluripotent stem cells. The scope of the invention encompasses all embodiments described herein, including variations in culture conditions, signaling modulators, cell surface markers, and downstream applications. Section headings are provided for organizational clarity and do not limit the interpretation of the claims. All references cited herein are incorporated by reference in their entirety for all purposes.

As used herein, the singular forms “a,” “an,” and “the” include plural referents unless the context clearly dictates otherwise. The term “about” means within ±10% of a stated value. “Nucleic acid” refers to deoxyribonucleic acid (DNA) or ribonucleic acid (RNA), including single-stranded, double-stranded, or partially double-stranded forms. A “polynucleotide” is a polymer of nucleotides, which may be natural or synthetic, linear or branched, and may include modified backbones or non-standard nucleotides. Nucleic acids may form duplexes via Watson-Crick base pairing, and may be labeled with detectable moieties such as fluorophores or enzymes for detection.

“Gene” refers to a DNA sequence encoding a functional product, typically a protein, though non-coding RNAs are also included. “Expression” denotes transcription and, where applicable, translation of a gene into its product. “Transcriptional regulatory sequences” include promoters, enhancers, silencers, and other cis-regulatory elements that modulate gene expression. A “promoter” is a region upstream of a transcription start site that directs RNA polymerase binding.

“Polypeptide,” “peptide,” and “protein” are used interchangeably to denote chains of amino acids linked by peptide bonds. “Fusion proteins” contain sequences from two or more distinct proteins. “Conservatively modified variants” refer to amino acid substitutions that preserve biochemical properties. Sequence identity is calculated using standard algorithms (e.g., BLAST) with default parameters.

Specific signaling molecules are defined as follows: “PDGFRα” is platelet-derived growth factor receptor alpha; “CXCR4” is C-X-C chemokine receptor type 4; “BMP” refers to bone morphogenetic proteins, particularly BMP4; “EGF” is epidermal growth factor; and “Wnt protein” denotes members of the Wnt family of secreted glycoproteins.

An “isolated” nucleic acid or protein is separated from its natural environment. A “cell” is a biological unit capable of metabolism and replication. A “primordial germ cell (PGC)” is an early germline progenitor specified during embryogenesis. The “epiblast” is the pluripotent embryonic layer preceding gastrulation; the “posterior epiblast” is its caudal region fated to give rise to PGCs and primitive streak. A “pluripotent stem cell” can differentiate into any somatic lineage but not extraembryonic tissues.

Cell culture is performed in defined media such as mTeSR1 or aRB27 basal medium, optionally supplemented with B27, non-essential amino acids, glutamine, and antibiotics. Serum-free conditions are preferred. Culture containers include multiwell plates coated with extracellular matrix proteins such as MATRIGEL. Conditions are maintained at 37°C, 5% CO₂, and ambient O₂ unless otherwise specified.

Vectors include plasmids and viral constructs for gene delivery. “Transfection” denotes non-viral nucleic acid delivery; “transduction” refers to viral-mediated transfer. “Contacting” means bringing cells into proximity with a compound or signal. An “inhibitor” reduces activity of a target pathway; an “agonist” enhances it. A “control” is a reference sample for comparison.

A “patient” is a human subject. “Treatment” includes therapeutic and prophylactic interventions. An “effective amount” is a dose that produces a desired biological response. “Administering” may occur via oral, parenteral, or other routes. Pharmaceutical compositions include carriers, excipients, and salts acceptable for human use.

### Methods

The invention provides a 2D in vitro platform for generating human PGCs from hPSCs, simplifying production for drug discovery, disease modeling, and infertility treatments. The method serves as a research tool to investigate lineage intermediates and extracellular signals in PGC specification.

PGC formation begins by contacting a pluripotent stem cell population with a WNT agonist (e.g., CHIR99021, 3 μM) and a TGFβ agonist (e.g., Activin-A, 100 ng/mL) for 12 hours in aRB27 basal medium, yielding a posterior epiblast cell population expressing MIXL1, BRACHYURY, and FGF8 alongside OCT4 and NANOG. The agonists are then removed, and the cells are contacted with a WNT inhibitor (e.g., XAV939, 1 μM) to form PGCs. This step is critical, as continued WNT signaling promotes mesoderm instead of germline fate.

PGCs are isolated via FACS using the surface marker profile CXCR4⁺PDGFRα⁻GARP⁻. “Separating” refers to physical isolation of this subpopulation. The pluripotent stem cell population may be hESCs or hiPSCs, maintained feeder-free on MATRIGEL in mTeSR1.

WNT agonists include CHIR99021, a GSK3β inhibitor; WNT inhibitors include XAV939, a tankyrase inhibitor that stabilizes AXIN. TGFβ agonists include Activin-A, which signals through SMAD2/3. Concentrations are optimized: CHIR99021 at 3 μM, Activin-A at 100 ng/mL, Y-27632 (ROCK inhibitor) at 10 μM to enhance survival.

After posterior epiblast induction, cells are expanded and contacted with BMP4 (40 ng/mL), SCF (100 ng/mL), and EGF (50 ng/mL) in the presence of XAV939 and Y-27632 for 72 hours total. BMP4 concentration is significantly lower than in 3D protocols (25-fold reduction), reflecting improved signal accessibility in monolayers.

Culture conditions are serum-free, at 37°C, 5% CO₂. Posterior epiblast markers include BRACHYURY, MIXL1, and NODAL. Human PSCs are used, including H1, H9, and multiple hiPSC lines. Contacting steps are sequential: first WNT/TGFβ activation, then WNT inhibition with BMP/SCF/EGF.

### Cell Compositions

The invention provides a cell culture composition comprising a WNT-activated posterior epiblast cell population, characterized by co-expression of pluripotency and primitive streak markers. Also provided is a purified PGC population with the surface phenotype CXCR4⁺PDGFRα⁻GARP⁻, cultured under serum-free conditions in aRB27 medium with XAV939, BMP4, SCF, and EGF.

### METHODS OF USE

PGCs generated by the disclosed methods may be used for infertility modeling, toxicology screening, epigenetic studies, and as a source for in vitro gametogenesis. They provide a human-relevant system to study germline development, imprinting disorders, and transgenerational inheritance.

### EXAMPLES

Primordial germ cells are essential for reproduction. Current PGC generation methods rely on inefficient 3D aggregates. The present invention introduces a simplified monolayer method. WNT activation for 12 hours induces posterior epiblast, which upon WNT inhibition specifies PGCs. Pluripotency factors OCT4 and NANOG are continuously expressed, bridging pluripotency and germline states. PGCs are purified using CXCR4⁺PDGFRα⁻GARP⁻ markers.

Human PGC origins are uncertain due to inaccessibility of early embryos. Mouse PGCs arise from posterior epiblast, but human data are limited. Previous methods used 3D cultures with high growth factor doses, yielding impure PGCLCs. The goal was a simplified 2D platform.

Primed PSCs are competent to form PGCs after brief posterior epiblast induction. Temporally dynamic WNT signaling is key: activation followed by inhibition. Unique surface markers enable purification. Single-cell RNA-seq reveals transcriptional trajectory.

Posterior epiblast cells bifurcate into PGCs or mesoderm. scRNA-seq shows PGCs express NANOS3, TFAP2C; non-PGCs express HAND1, ACTC1. WNT inhibition boosts PGC yield 3-fold. BMP4 at 40 ng/mL suffices in monolayers.

Pure PGCs form within 3.5 days. They express PGC markers and lack somatic markers. The CXCR4⁺PDGFRα⁻GARP⁻ profile is validated across hESC/hiPSC lines. Efficiency reaches 73.2% in some lines.

NANOG-2A-YFP reporter confirms continuous NANOG expression. scRNA-seq shows in vitro PGCs resemble fetal PGCs (FGC1 stage). Materials and methods detail culture, differentiation, FACS, and sequencing protocols.

## P EMBODIMENTS

The invention defines a PGC formation method comprising: (a) contacting hPSCs with WNT and TGFβ agonists for 12 hours; (b) removing agonists; (c) contacting with WNT inhibitor and BMP/SCF/EGF for 72 hours. PGC isolation uses CXCR4⁺PDGFRα⁻GARP⁻ FACS. Duration of initial contacting is 12±2 hours. PGC formation completes by day 3.5. Pluripotent stem cells are expanded pre-differentiation. Additional factors include Y-27632 (10 μM).

### EMBODIMENTS

A PGC formation method involves temporal WNT modulation. Isolation uses CXCR4⁺PDGFRα⁻GARP⁻. Initial contacting lasts 12 hours. PGCs form by 84 hours. hPSCs are expanded in mTeSR1. Additional factors: Y-27632, BMP4 (40 ng/mL), SCF (100 ng/mL), EGF (50 ng/mL). Culture conditions: 37°C, 5% CO₂, serum-free aRB27 medium. PGCs express NANOS3, PRDM1, TFAP2C, SOX17, OCT4, NANOG. Medium contains B27, NEAA, glutamine, pen/strep. Infertility treatment involves generating patient-specific PGCs for gamete derivation.