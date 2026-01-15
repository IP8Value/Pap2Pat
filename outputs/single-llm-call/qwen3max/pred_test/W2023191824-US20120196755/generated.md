# DESCRIPTION

## FEDERALLY SPONSORED RESEARCH

This invention was made with government support under Grant No. R01 GM084293 awarded by the National Institutes of Health. The government has certain rights in the invention.

## TECHNICAL FIELD

The present invention relates to the field of molecular genetics and genomic analysis, particularly to methods for the direct, genome-wide mapping of chromosomal fragile sites through the detection of single-stranded DNA (ssDNA) and double-stranded breaks (DSBs). The invention further encompasses compositions, kits, systems, and computer-readable media for implementing such methods, as well as their application in the diagnosis, monitoring, and therapeutic management of chromosomal breakage syndromes, cancer, and other conditions associated with replication stress and genomic instability.

## BACKGROUND

Chromosomal breakage syndromes are a group of inherited disorders characterized by elevated levels of spontaneous or induced chromosomal instability, including gaps, breaks, and rearrangements, often resulting from defects in the cellular response to DNA replication stress. These syndromes are associated with increased cancer predisposition, developmental abnormalities, and immunodeficiency. Understanding the molecular origins of chromosomal fragility is critical for elucidating disease mechanisms and developing diagnostic and therapeutic strategies.

Chromosomal breakage arises from a variety of endogenous and exogenous sources that impede the progression of DNA replication forks. When replication forks encounter obstacles such as DNA lesions, secondary structures, tightly bound proteins, or nucleotide pool depletion, they may stall or collapse, leading to the formation of ssDNA regions and, ultimately, DSBs. Such events are particularly prevalent at so-called chromosome fragile sites (CFSs), which are genomic regions inherently prone to breakage under conditions of replication stress.

Chromosome fragile sites have been classified as common or rare based on their frequency in the population. Common fragile sites (CFSs) are present in all individuals and become cytogenetically visible upon treatment with low doses of aphidicolin or hydroxyurea (HU), agents that induce replication stress by inhibiting DNA polymerases or depleting nucleotide pools, respectively. Despite their clinical relevance, the precise molecular features that define CFSs and render them susceptible to breakage have remained elusive, in part due to limitations in existing mapping technologies.

Replication fork progression is a highly coordinated process involving the assembly of numerous protein complexes that ensure accurate and efficient DNA synthesis. In the presence of replication stress, the replication checkpoint—mediated by kinases such as Mec1 (ATR in humans) and Rad53 (CHK2 in humans)—is activated to stabilize stalled forks, prevent premature origin firing, and allow time for repair. In the absence of a functional checkpoint, as in chromosomal breakage syndromes like Ataxia-telangiectasia or Fanconi anemia, stalled forks are prone to collapse, leading to DSBs and genomic rearrangements.

Accurate mapping of ssDNA and DSBs across the genome is essential for identifying fragile sites and understanding the dynamics of replication fork progression and collapse. However, prior methods have suffered from significant limitations, including indirect detection, reliance on protein markers that may not be stably associated with breaks, susceptibility to in vitro artifacts during DNA isolation, and inability to distinguish between true DSBs and internal ssDNA regions. These shortcomings have hindered the development of reliable predictive models for chromosomal fragility.

There exists a long-felt but unmet need for a robust, direct, and high-resolution method capable of mapping both ssDNA and DSBs genome-wide in a manner that minimizes in vitro artifacts and provides quantitative, spatially resolved information about sites of replication stress and chromosomal breakage. Such a method would enable the identification of novel fragile sites, the characterization of replication dynamics in health and disease, and the development of diagnostic tools for chromosomal breakage disorders and cancer.

## SUMMARY

The present invention provides a novel, genome-wide method for the direct detection and mapping of chromosomal fragile sites by simultaneously labeling and quantifying single-stranded DNA (ssDNA) and double-stranded breaks (DSBs) in genomic DNA. The method overcomes the limitations of prior art by performing labeling reactions directly within an agarose matrix that stabilizes the cellular milieu and minimizes in vitro DNA damage during sample preparation.

In one aspect, the invention provides a method comprising embedding cells in a solid matrix, such as agarose, followed by in situ lysis to remove cell walls or membranes and degrade proteins while preserving the integrity of genomic DNA. This step is critical for preventing artifactual DNA breaks that commonly occur during conventional DNA isolation procedures.

The method then involves directly labeling ssDNA regions in the intact genomic DNA embedded within the matrix. This is achieved by incubating the matrix with a template-dependent DNA polymerase, such as exonuclease-deficient Klenow fragment or Sequenase, along with a mixture of unlabeled deoxynucleotides and a labeled deoxynucleotide, typically a fluorescently tagged dUTP (e.g., Cy3-dUTP or Cy5-dUTP) or a biotinylated dUTP. Random primers, such as random hexamers, are included to prime synthesis at ssDNA regions, thereby incorporating the label specifically into these sites.

Concurrently or separately, the method enables the direct labeling of DSBs by incubating the matrix with a mixture of enzymes capable of processing and labeling DNA ends. In a preferred embodiment, this enzyme mixture comprises T4 DNA polymerase and polynucleotide kinase, as found in commercially available kits such as the End-It DNA End-Repair Kit. The T4 DNA polymerase utilizes its 3′-5′ exonuclease and polymerase activities to generate blunt ends from either 3′ or 5′ overhangs, while simultaneously incorporating labeled nucleotides at the break sites. The polynucleotide kinase phosphorylates 5′ ends, although this activity is not essential for labeling but does not interfere with the process.

Following labeling, the labeled DNA is isolated from the matrix by electroelution or enzymatic digestion of the agarose (e.g., using agarase). The eluted DNA may optionally be fragmented to a desired size range, typically 200–1000 base pairs, using sonication or enzymatic methods, to facilitate downstream analysis.

The labeled DNA is then separated from non-labeled DNA using affinity purification if an affinity label such as biotin is used. For example, streptavidin-coated magnetic beads or columns can be employed to capture biotinylated DNA fragments corresponding to ssDNA or DSB sites. If fluorescent labels are used, separation may be omitted, and the labeled DNA can be directly processed for detection.

Detection of the labeled DNA is performed using high-throughput sequencing or microarray hybridization. In a preferred embodiment, the labeled DNA is prepared for next-generation sequencing by ligating platform-specific adapters, performing PCR amplification if necessary, and immobilizing the library on a solid support for sequencing. Sequencing techniques may include reversible terminator sequencing (e.g., Illumina), pyrosequencing (e.g., 454), sequencing by ligation (e.g., SOLiD), or real-time single-molecule sequencing (e.g., PacBio or Oxford Nanopore). Alternatively, the labeled DNA may be co-hybridized with a control sample to a DNA microarray, and the relative enrichment of label at each genomic locus is determined by fluorescence ratio analysis.

The invention is applicable to a wide range of cell types, including yeast, mammalian cells, and primary patient samples. The matrix-based approach ensures that the genomic architecture is preserved during labeling, enabling accurate correlation between ssDNA formation and subsequent DSB occurrence. The method reveals that ssDNA accumulates at stalled replication forks prior to breakage and that the location of ssDNA predicts the eventual site of DSB, providing a powerful predictive tool for identifying fragile sites.

The benefits of the invention include high specificity, minimal background from in vitro artifacts, compatibility with both ssDNA and DSB detection in the same experimental framework, and scalability to high-throughput applications. The method enables the generation of high-resolution maps of replication stress and chromosomal fragility, facilitating research into DNA repair mechanisms, replication dynamics, and the etiology of genomic instability disorders.

## DETAILED DESCRIPTION

The following detailed description is provided to enable any person skilled in the art to make and use the invention. Specific details are set forth to provide a thorough understanding of the invention, but it will be apparent that the invention may be practiced without these details. Well-known methods, procedures, and components have not been described in excessive detail to avoid obscuring the invention.

The invention encompasses multiple embodiments, all of which fall within the scope of the claims. As used herein, the singular forms “a,” “an,” and “the” include plural referents unless the context clearly dictates otherwise. All publications, patents, and patent applications cited herein are hereby incorporated by reference in their entirety for all purposes.

A central discovery underlying the present invention is the predictive relationship between ssDNA formation and subsequent DSB occurrence at chromosomal fragile sites. Specifically, ssDNA accumulates at stalled replication forks under conditions of replication stress, and this ssDNA persists in checkpoint-deficient cells, ultimately leading to DSBs at the same genomic locations. This temporal and spatial correlation allows ssDNA mapping to serve as an early indicator of potential breakage sites.

The invention introduces a novel approach to directly map sites of chromosome fragility by performing labeling reactions in situ within a stabilizing matrix. This matrix, preferably low-melting-point agarose, encapsulates whole cells and maintains the native conformation of chromosomal DNA during lysis and labeling, thereby preventing mechanical shearing and nuclease-mediated degradation that plague conventional DNA isolation methods.

Genomic DNA preparation begins with embedding cells in agarose plugs. The plugs are then treated with enzymes such as zymolyase or lyticase (for yeast) or detergents and proteases (for mammalian cells) to remove cell walls or membranes and digest proteins, leaving high-molecular-weight DNA entrapped in the agarose. This step is conducted under mild conditions to preserve DNA integrity.

Labeling of ssDNA is performed by incubating the agarose plug with a reaction mixture containing a template-dependent DNA polymerase that lacks 3′-5′ exonuclease activity (to prevent degradation of ssDNA templates), random primers (e.g., hexamers or octamers), a balanced mix of unlabeled dNTPs, and a detectably labeled dUTP. The polymerase extends the random primers only where ssDNA is present, incorporating the label specifically at these sites. Control experiments using polymerases with and without strand-displacement activity confirm that labeling reflects genuine ssDNA gaps rather than nicks or displaced strands.

For DSB labeling, the agarose plug is incubated with an enzyme mixture capable of recognizing and labeling free DNA ends. A preferred mixture includes T4 DNA polymerase, which can fill in 5′ overhangs and chew back 3′ overhangs while incorporating labeled nucleotides, and polynucleotide kinase, which phosphorylates 5′ ends. Commercially available end-repair kits, such as the End-It kit, are suitable for this purpose. The reaction is optimized to ensure efficient labeling of diverse end structures while minimizing non-specific incorporation.

The choice of label is flexible. Fluorescent labels (e.g., Cy3, Cy5, Alexa Fluor dyes) enable direct detection via microarray or imaging. Affinity labels such as biotin allow purification using streptavidin-conjugated beads or columns, which is particularly useful for sequencing applications where background reduction is critical. Other labels, including digoxigenin or haptens, may also be used with appropriate detection reagents.

After labeling, DNA is eluted from the agarose matrix by electroelution or agarase digestion. The eluted DNA may be fragmented to a uniform size (e.g., 500 bp) by sonication to improve resolution in downstream analyses. If biotinylated, the DNA is captured on streptavidin beads, washed stringently, and eluted or used directly for adapter ligation.

For sequencing, standard library preparation steps are followed: end repair (if not already performed), A-tailing, adapter ligation, and PCR amplification. The resulting library is sequenced using any high-throughput platform. Sequence reads are aligned to a reference genome, and enrichment peaks are identified using bioinformatic algorithms to map ssDNA and DSB locations.

Prior methods for mapping fragile sites suffer from significant drawbacks. Indirect end-labeling requires Southern blotting and is low-throughput. Tandem PCR is biased and limited to known regions. Microarray-based ChIP-chip relies on protein binding, which may not occur at all breaks. ssDNA enrichment using benzoyl-napthoyl-DEAE-cellulose cannot distinguish internal ssDNA from DSB-associated ssDNA. γ-H2AX detection marks DSBs but also responds to other stresses and may yield false positives. The present invention overcomes these limitations by directly labeling the DNA lesions themselves in a protected environment.

The method has been validated in yeast models. In mec1 checkpoint-deficient cells treated with HU, ssDNA accumulates near late-firing origins during HU exposure and persists after HU removal. DSBs subsequently form at these same locations during recovery. In contrast, wild-type cells show minimal breakage. The method accurately maps restriction enzyme-induced breaks and HO endonuclease cuts, confirming its specificity. Importantly, internal ssDNA regions in HU-treated cells do not produce significant DSB signals, demonstrating that the end-labeling reaction is specific for true DNA ends.

Application to mammalian genomes follows the same principles. Human cells are embedded in agarose, lysed in situ, and subjected to ssDNA and DSB labeling. The method can identify common fragile sites such as FRA3B and FRA16D under aphidicolin treatment and reveal novel sites associated with oncogene activation or chemotherapy.

The invention has broad applications in the study of chromosomal breakage disorders. Ataxia-telangiectasia, caused by ATM mutations, exhibits radiosensitivity and chromosomal instability. Bloom syndrome, due to BLM helicase deficiency, shows elevated sister chromatid exchanges. Xeroderma pigmentosum involves defective nucleotide excision repair. Nijmegen breakage syndrome results from NBS1 mutations affecting the MRN complex. Fanconi anemia, characterized by crosslinker sensitivity, can lead to bone marrow failure and leukemia. The method can diagnose these conditions by quantifying baseline and stress-induced fragility, monitor disease progression, and assess therapeutic responses.

The method also enables high-throughput screening of clastogenic agents. Cells are exposed to test compounds, embedded in agarose, and processed for DSB labeling. Compounds inducing significant breakage profiles are flagged as potential genotoxins. This approach is faster and more comprehensive than traditional cytogenetic assays.

Kits for mapping CFS are provided, comprising agarose, lysis reagents, labeling enzymes (Klenow, T4 DNA polymerase), labeled nucleotides, random primers, buffers, and instructions. Optional components include streptavidin beads, sequencing adapters, and control DNA.

For monitoring treatment efficacy, particularly in cancer, patient cells are sampled before and during therapy. Reduced ssDNA/DSB signals after treatment indicate effective suppression of replication stress or enhanced DNA repair, guiding dose adjustment or drug selection.

The term “subject” refers to any human or animal patient undergoing evaluation or treatment for a condition involving genomic instability.

Systems for mapping CFS include robotic liquid handlers for plug preparation, thermal cyclers for labeling, electrophoresis units for elution, sonicators, bead separators, and sequencing platforms. Integrated software controls workflows and analyzes data.

Computer-readable storage media store executable code for aligning sequence reads, calling peaks, correlating ssDNA and DSB profiles, and generating reports. The media may be cloud-based or local.

Variations of systems and media may incorporate machine learning to predict fragility from sequence features or integrate with electronic health records.

Advantages include unprecedented accuracy, directness, versatility across species, and compatibility with clinical samples. The scope encompasses all combinations of the described elements, including hybrid methods using both fluorescence and affinity labels.

In conclusion, the invention provides a transformative platform for studying and diagnosing chromosomal fragility, with far-reaching implications for basic research and precision medicine.

## EXAMPLES

### Example 1

The invention was demonstrated in the yeast Saccharomyces cerevisiae using strains HM14-3a (wild-type RAD53), WFA34 (rad53K227A mutant), BY2006 (mec1-1), RCY378 (mec1-4), RCY301 (MEC1 control), and YSCL004 (HO endonuclease-inducible). Cells were grown at 30°C in synthetic complete medium. For synchronization, bar1 strains were treated with 200 nM α-factor and BAR1 strains with 3 μM α-factor; pronase was added to degrade α-factor (25 μg/ml for bar1, 300 μg/ml for BAR1). Hydroxyurea (HU) was used at 200 mM to induce replication stress.

Cells were embedded in agarose plugs and spheroplasted using standard CHEF gel protocols. For ssDNA labeling, plugs were pre-equilibrated in Tris-EDTA, then in labeling buffer (50 mM Tris-HCl pH 6.8, 5 mM MgCl2, 10 mM β-mercaptoethanol). Each plug slice containing ~10^8 cells was incubated with 50 μl labeling mix: 0.24 mM dATP/dCTP/dGTP, 0.12 mM dTTP, 0.12 mM Cy5-dUTP, 250 μg/ml random hexamers, and 150 units Klenow exo– at 37°C for 2 hours in the dark. Labeled DNA was electroeluted in dialysis tubing at 110 V for 3 hours, sonicated to ~500 bp, and purified.

For DSB labeling, plugs were equilibrated in End-Repair buffer, then incubated with End-Repair mix: 1 mM ATP, 0.24 mM dATP/dCTP/dGTP, 0.12 mM dTTP, 0.12 mM Cy3-dUTP, and 3 μl End-Repair enzyme mix at room temperature for 1 hour. Electroelution and processing were identical to ssDNA labeling.

Experimental and control DNAs (differentially labeled) were co-hybridized to Agilent yeast 4x44K microarrays. Data were extracted, normalized, and smoothed with a 6 kb Lowess window. Peaks were identified as significant breakage or ssDNA sites.

As a control, BamHI digestion of DNA in plugs produced a breakage profile matching known BamHI sites, validating the method. Similarly, HO induction in YSCL004 yielded breaks at expected loci.

In mec1 cells, ssDNA formed near late-firing origins during HU treatment and persisted after HU removal. DSBs occurred at these same sites during recovery, with a correlation coefficient of 0.64 between ssDNA and breakage profiles. Replication fork progression experiments (T0-R vs. T20-R) showed that breakage sites shifted with fork movement, confirming that breakage correlates with fork position, not origin identity. Random simulation tests confirmed significant association between breakage and checked origins (P < 0.0001).

### Example 2

Human HeLa cells were cultured in DMEM with 10% FBS. Aphidicolin, a DNA polymerase inhibitor, was added at 0.2 μM for 24 hours to induce replication stress. Cells were harvested, washed, and embedded in 1% low-melting agarose. Plugs were lysed in buffer containing 1% Sarkosyl, 1 mg/ml proteinase K, 50 mM EDTA, pH 8.0, at 50°C for 48 hours, then washed extensively.

For ssDNA labeling, plugs were incubated with Klenow fragment (exo–), random hexamers, dNTPs, and biotin-dUTP in labeling buffer at 37°C for 2 hours. For DSB labeling, plugs were treated with End-Repair enzyme mix and biotin-dUTP at room temperature for 1 hour.

Agarose was digested with agarase, and DNA was sonicated. Biotinylated DNA was captured on streptavidin beads, washed, and eluted. Libraries were prepared by end repair, A-tailing, adapter ligation, and PCR. Sequencing on an Illumina platform revealed enrichment at known common fragile sites (e.g., FRA3B), demonstrating the method’s applicability to mammalian genomes and its utility in identifying origins of replication and fragile regions.