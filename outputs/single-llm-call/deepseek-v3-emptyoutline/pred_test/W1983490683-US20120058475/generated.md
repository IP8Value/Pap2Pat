Here is the complete patent application following the provided outline:

## FIELD OF THE INVENTION

The present invention relates generally to the field of cytogenetics and molecular biology, and more specifically to novel methods for detecting chromosomal inversions with high resolution and specificity. The invention encompasses a directional genomic hybridization methodology termed "chromatid painting" that enables strand-specific detection of chromosomal inversions at unprecedented resolution. This technology represents a significant advancement over traditional cytogenetic approaches by combining the whole-genome visualization capabilities of classical cytogenetics with the sequence specificity of molecular techniques. The methodology is particularly useful for identifying chromosomal inversions associated with various disease states, including but not limited to cancer, developmental disorders, and idiopathic infertility. Furthermore, the invention has applications in evolutionary biology, comparative genomics, and toxicological risk assessment by providing a means to detect chromosomal inversions induced by genotoxic agents.

## BACKGROUND OF THE INVENTION

Chromosomal inversions represent a class of structural rearrangements where a segment of DNA is reversed end-to-end within a chromosome. While these rearrangements have been recognized for decades, their detection and characterization have remained challenging due to limitations inherent in existing methodologies. Traditional cytogenetic approaches relying on banding patterns can only detect inversions larger than approximately 10 megabases, while molecular techniques such as fluorescence in situ hybridization (FISH) are generally incapable of detecting orientation changes in DNA sequences. More contemporary approaches including microarray-based comparative genomic hybridization (a-CGH) and next-generation sequencing have expanded our understanding of chromosomal abnormalities, yet these methods still face significant limitations in detecting and characterizing inversions.

The difficulty in detecting inversions stems from several fundamental challenges. First, traditional banding analysis relies on visual pattern recognition of megabase-sized striations along chromosomes, which severely limits resolution and often fails to reveal inversions that don't visibly alter banding patterns. Second, while whole chromosome painting by FISH can identify inter-chromosomal rearrangements, it cannot detect intra-chromosomal changes in orientation. Third, sequencing-based approaches, while powerful for identifying sequence variations, struggle with phase determination and are particularly challenged by the presence of flanking inverted duplications or mis-assemblies in reference genomes. Additionally, sequencing methods are poorly suited for analyzing cellular heterogeneity in solid tumors or for constructing quantitative cellular dose responses following exposure to genotoxic agents.

Prior attempts to address these limitations have included chromosome orientation FISH (CO-FISH), which was limited to interrogating repetitive centromeric or telomeric sequences with simple oligonucleotide probes. While this approach demonstrated the potential of strand-specific hybridization, its utility was constrained by the repetitive nature of target sequences and the inability to analyze unique sequences along chromosome arms. There remains therefore a significant unmet need for methodologies that combine the whole-genome visualization capabilities of cytogenetics with the high resolution of molecular approaches, particularly for detecting chromosomal inversions at scales ranging from kilobases to megabases.

## SUMMARY OF THE INVENTION

The present invention provides a novel methodology termed "chromatid painting" that enables high-resolution detection of chromosomal inversions through directional genomic hybridization. This technique overcomes the limitations of prior approaches by combining strand-specific hybridization with bioinformatics-guided probe design to achieve sequence-specific detection of orientation changes along entire chromosomes or targeted regions thereof.

The methodology involves several key innovations. First, cells are cultured with bromo-deoxyuridine/deoxycytidine (BrdU/BrdC) during a single round of replication to create unifilarly substituted sister chromatids. Following appropriate pretreatment to render chromatids single-stranded, strand-specific oligonucleotide probes designed to have uniform 5'→3' directionality are hybridized to their complementary targets. The probes are designed using bioinformatics approaches to target unique sequences at defined intervals along chromosomes, typically spaced at approximately 1 megabase intervals, though closer spacing is possible for higher resolution. When an inversion is present, the obligatory reversal of DNA strand orientation causes the hybridization signal to switch from one sister chromatid to the other, providing a clear cytogenetic signature of the inversion.

The invention encompasses several embodiments including: (1) whole-chromosome chromatid paints consisting of thousands of individual strand-specific oligonucleotides that provide genome-wide inversion screening capability; (2) targeted probe sets designed to interrogate specific regions of interest, such as known inversion breakpoints associated with particular diseases; (3) methods for detecting inversions as small as a few kilobases through dense probe coverage of specific regions; and (4) combined approaches that simultaneously detect both intra-chromosomal inversions and inter-chromosomal rearrangements such as translocations.

The chromatid painting methodology offers several advantages over existing techniques. It provides resolution that is approximately an order of magnitude better than traditional banding analysis, with the potential to detect inversions as small as a few kilobases when using appropriately spaced probes. Unlike sequencing-based approaches, it maintains the ability to analyze individual cells, making it particularly valuable for studying cellular heterogeneity in tumors or for dose-response assessments following genotoxic exposure. Furthermore, the technique can be applied to any species with a sequenced genome, enabling comparative genomic studies across diverse organisms.

## DETAILED DESCRIPTION OF THE INVENTION

The present invention provides comprehensive methods for detecting chromosomal inversions through directional genomic hybridization, termed chromatid painting. This technology represents a significant advancement in cytogenetic analysis by enabling strand-specific detection of orientation changes in chromosomal DNA at resolutions ranging from kilobases to megabases. The detailed methodology encompasses several key components and steps that together provide a robust system for inversion detection and characterization.

### Selection of Sequences:

The foundation of chromatid painting lies in the careful selection and design of strand-specific oligonucleotide probes. The process begins with downloading contiguous DNA sequences from genomic databases such as the NCBI genomic database (GRCh37.p2 primary assembly in the exemplary embodiment). These sequences are then processed to mask or remove repetitive elements using software such as that provided by the Genetic Information Research Institute. This step is crucial for ensuring probe specificity to unique sequences.

Single-stranded oligonucleotide probes are designed to target these unique sequences with specific characteristics. In the exemplary embodiment, probes are typically 40-mers designed to have uniform melting temperatures (70.0±5.0°C) and consistent 5'→3' directionality. The probes are designed to tile across target regions at specified non-overlapping locations along chromosome contigs, typically at approximately 1 megabase intervals, though this spacing can be adjusted based on the desired resolution. Probe design utilizes specialized software such as ARRAY Designer (version 4.2) or proprietary software (KromaTiD Inc. in the exemplary embodiment) to ensure optimal performance characteristics.

The selection process emphasizes several key parameters: (1) probe length and melting temperature uniformity to ensure consistent hybridization behavior across all probes in a set; (2) absolute strand specificity to guarantee hybridization only to the intended single-stranded target; (3) genomic distribution to provide comprehensive coverage of the target chromosome or region; and (4) absence of cross-hybridization potential to non-target sequences. These characteristics are essential for achieving the high specificity and sensitivity required for reliable inversion detection.

### EXAMPLE 2

An exemplary embodiment of the chromatid painting methodology involves the creation and application of a human chromosome 3-specific chromatid paint. This example demonstrates the process from probe design through hybridization and analysis, illustrating the key steps and outcomes of the methodology.

The process begins with selection of the largest contig on chromosome 3q (NT_005612 in the exemplary embodiment) as the initial target. Probe sets to unique sequences within this contig are designed, synthesized, and labeled with fluorescent markers (green fluorescein in this example). These initial probe sets are hybridized to pre-treated human metaphase chromosomes to validate strand specificity and confirm signal at the predicted chromosomal locations.

Following successful validation with the initial contig, the process is extended to the remaining three contigs of chromosome 3. For these contigs, probe sets containing 90 individual oligonucleotides each are designed, synthesized, and labeled with a different fluorochrome (red Cy3 in this example). Hybridization of these additional probe sets confirms strand-specific signals at their predicted locations along chromosome 3, with all signals appearing on the same chromatid. This validation confirms similar 5'→3' directionality of the assembled contigs in the reference genome.

The complete chromosome 3 chromatid paint is then constructed by synthesizing and pooling approximately 17,000 individual single-stranded oligonucleotides designed to cover the entire chromosome at approximately 1 megabase intervals. These oligonucleotides are fluorescently labeled and applied to pre-treated metaphase chromosome spreads. The resulting hybridization demonstrates robust and specific painting of one chromatid of chromosome 3, with no detectable signal on the sister chromatid or other chromosomes in the spread.

This example further demonstrates the application of chromatid painting for genome-wide inversion discovery. By examining cells exposed to ionizing radiation (a known inducer of chromosomal inversions), both small and large inversions are readily detected as abrupt switches in hybridization signal from one sister chromatid to the other. The resolution of detection is directly related to the spacing of probe sets, with inversions of 1 megabase or larger reliably detected by the 1 megabase-spaced probe sets. Smaller inversions may also be detected if they happen to include a probe target sequence, with the probability of detection decreasing with inversion size.

### EXAMPLE 3

Another exemplary embodiment demonstrates the application of chromatid painting for high-resolution targeted inversion detection at specific chromosomal loci of clinical significance. This example illustrates the methodology's utility in detecting known inversion breakpoints associated with specific diseases.

The first application involves analysis of the Kasumi-4 cell line, derived from a patient with chronic myelogenous leukemia (CML) and known to possess a chromosome 3 homolog with a large q21;q26 inversion. Targeted strand-specific probe sets are designed to sequences flanking both sides of the known inversion breakpoints (within 1 megabase in this example) and labeled with different fluorochromes. Hybridization of these probe sets to metaphase chromosomes from the Kasumi-4 cells readily identifies the inverted homolog as the chromosome where one fluorochrome signal (red in this example) switches to the opposite chromatid, while the other fluorochrome signal (green) remains on the original chromatid, representing sequences proximal and distal to the inversion breakpoints.

The second application demonstrates detection of the RET/PTC1 rearrangement associated with radiation-induced papillary thyroid carcinoma, which involves a q11.2;q21 inversion in chromosome 10. HTori-3 immortalized human thyroid cells, some of which contain this inversion, are analyzed with strand-specific probe sets targeted to the inversion region on chromosome 10. The analysis reveals cells containing the inv(10)(q11.2;q21) rearrangement through clear splitting of strand-specific signals across sister chromatids, with one signal remaining on the original chromatid and the other switching to the sister chromatid within the inverted segment.

This example further illustrates the detection of very small inversions through dense probe coverage. A mock "mini-inversion" is created within a 10 megabase region of chromosome 3q by designing probe sets that cover the entire region with fluorescein-labeled oligos, except for a specifically excluded 6 kilobase segment. For this excluded segment, oligos are intentionally designed in the reverse orientation and labeled with Cy3. Hybridization reveals the simulated 6 kilobase inversion as a small red signal on the opposite chromatid from the predominant green signal, demonstrating the potential for detecting inversions at the kilobase scale when using appropriately designed probe sets.

### Hybridization:

The hybridization process in chromatid painting represents a critical and innovative component of the invention. The methodology involves several carefully controlled steps to ensure specific and sensitive detection of chromosomal inversions.

Prior to hybridization, cells must be appropriately prepared. Cells are cultured for a single cell cycle in media containing 5.0 μM 5-bromo-2-deoxyuridine and 1.0 μM 5-bromo-deoxycytidine (BrdU/BrdC) to achieve unifilar substitution of DNA strands. Mitotic cells are collected following Colcemid treatment and prepared on slides using standard cytogenetic protocols.

The hybridization process begins with slide pretreatment to render chromatids single-stranded. Slides are incubated in PN buffer (sodium phosphate) for 10 minutes at room temperature, rinsed, and dehydrated through an ethanol series. They are then stained with Hoechst 33258, exposed to 365 nm ultraviolet light to nick the DNA at BrdU incorporation sites, and treated with Exonuclease III to selectively degrade the newly replicated strands. This pretreatment effectively converts each sister chromatid into a single-stranded target for subsequent hybridization.

For hybridization, a mixture containing hybridization buffer, chromatid paint (the pooled strand-specific probes), and water is prepared and heated to 75°C for 5 minutes to denature the probes. This mixture is applied to pretreated slides, which are coverslipped, sealed, and heated at 73°C for 3 minutes to denature target DNA. Slides are then transferred to hybridization chambers and incubated at 37°C overnight to allow probe hybridization.

Following hybridization, slides are washed five times in 2× SSC at 42°C to remove unbound probes, rinsed in PN buffer, and counterstained with DAPI/antifade for visualization. The entire process typically achieves hybridization efficiencies greater than 90%, as assessed by signal presence at expected chromosomal locations.

### Detection:

The detection of chromosomal inversions using chromatid painting relies on the principle that inverted DNA segments must reverse their 5'→3' orientation to maintain polarity. This obligatory reversal causes individual strands within the inversion to "switch places" between sister chromatids, resulting in a microscopically visible signal switch when hybridized with directional single-stranded probes.

Detection is performed using fluorescence microscopy with appropriate filter sets for the fluorochromes used in probe labeling. In the exemplary embodiment, an Olympus Bx41 microscope equipped with fluorochrome-specific excitation/barrier filters and a Photometrics CoolSNAP ES 2 camera running Metavue 7.1 software is used. However, any comparable fluorescence microscopy system with appropriate capabilities could be employed.

The detection process involves several key observations and interpretations:
1. For large inversions (≥1 megabase with standard probe spacing), the signal switch from one chromatid to its sister is accompanied by a corresponding lack of signal on the opposite chromatid in the inverted region.
2. For smaller inversions, the unlabeled segment may be obscured by the brightness of adjacent fluorescent signals, making the signal switch the primary detectable feature.
3. In targeted analyses of known inversion breakpoints, the relative positions of differently colored probes (representing sequences flanking the breakpoints) provide additional confirmation of inversion presence and orientation.
4. The intensity and continuity of signals may vary depending on the state of chromatin condensation, providing potential information about chromosome structure beyond simple inversion detection.

The detection methodology is sufficiently sensitive to identify inversions present in only a subset of cells within a population, making it valuable for studying mosaic conditions such as those found in tumors or following genotoxic exposure.

### Results:

Application of chromatid painting methodology yields several important results that demonstrate its utility and advantages over existing techniques.

In validation studies using the chromosome 3 chromatid paint, the methodology demonstrates:
1. Robust and specific hybridization confined to one chromatid of the target chromosome, with efficiency routinely exceeding 90%.
2. Confirmation of reference genome assembly by verifying consistent 5'→3' directionality of contigs along the entire length of chromosome 3.
3. Detection of both small and large inversions induced by ionizing radiation, with inversion size directly related to probe spacing density.
4. Successful detection of a simulated 6 kilobase inversion through specially designed probe sets, demonstrating the potential for very high-resolution analysis.

In applications to known disease-associated inversions, the methodology successfully:
1. Identifies the chromosome 3 q21;q26 inversion in Kasumi-4 CML cells through clear signal switching between chromatids at the inversion breakpoints.
2. Detects the RET/PTC1 rearrangement (inv(10)(q11.2;q21)) in irradiated HTori-3 thyroid cells, confirming the technology's utility for studying radiation-associated chromosomal changes.
3. Reveals a low spontaneous inversion frequency (<0.3%) in normal human fibroblasts and lymphocytes, establishing a baseline for assessing induced inversions.

Additional significant results include:
1. Successful cross-species hybridization to chimpanzee, gorilla, and orangutan chromosomes, demonstrating utility for comparative genomic studies.
2. Detection of a large inversion in both homologs of orangutan chromosome 3, highlighting applications in evolutionary biology.
3. Simultaneous detection of intra-chromosomal inversions and inter-chromosomal rearrangements (e.g., dicentrics and translocations) in the same cells.
4. Confirmation that strand-specific probes maintain their specificity when used for standard (double-stranded) FISH applications, expanding potential diagnostic uses.

These results collectively demonstrate that chromatid painting provides a powerful new tool for high-resolution inversion detection with applications spanning basic research, clinical diagnostics, toxicological assessment, and evolutionary studies. The methodology's ability to detect inversions at scales ranging from kilobases to megabases, combined with its whole-genome visualization capability at single-cell resolution, represents a significant advance over existing technologies.