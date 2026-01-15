# DESCRIPTION

## RELATED APPLICATION DATA

This application claims the benefit of U.S. Provisional Patent Application No. 63/456,789, filed on October 12, 2022, which is hereby incorporated by reference in its entirety for all purposes.

## STATEMENT OF GOVERNMENT INTERESTS

This invention was made with government support under Grant No. HG012345 awarded by the National Institutes of Health. The government has certain rights in the invention.

## FIELD

The present invention relates generally to methods of making proteins containing non-standard amino acids (NSAAs) and, more particularly, to in vivo systems that employ post-translational proofreading mechanisms to selectively degrade misincorporated or undesired amino acid variants while enriching for target polypeptides bearing desired NSAAs at defined positions. The invention further encompasses genetically engineered cells, orthogonal translation systems, and engineered components of the N-end rule pathway to enhance fidelity and yield of NSAA-containing protein production.

## BACKGROUND

Proteins are naturally synthesized from a set of twenty standard amino acids, which are encoded by the canonical genetic code and incorporated into nascent polypeptide chains during ribosomal translation. Over the past several decades, significant advances have been made in expanding the genetic code to include non-standard amino acids (NSAAs)—unnatural or chemically modified amino acids that possess functionalities not found in the natural repertoire. These NSAAs enable novel chemical, physical, and biological properties in engineered proteins, facilitating applications in therapeutics, diagnostics, materials science, and synthetic biology.

Incorporation of NSAAs into proteins typically relies on orthogonal translation systems (OTS), which consist of an orthogonal aminoacyl-tRNA synthetase (aaRS) and its cognate tRNA that do not cross-react with endogenous host machinery. These OTS components are engineered to specifically charge the tRNA with a desired NSAA and deliver it to the ribosome in response to a reassigned codon, most commonly the amber stop codon (UAG). Despite these advances, a major limitation persists: many OTSs exhibit promiscuity, leading to misincorporation of standard amino acids or structurally similar NSAAs at the target site. This reduces the purity and functional homogeneity of the resulting protein product, which is particularly problematic for applications requiring high fidelity, such as biocontainment, biologics manufacturing, or precision enzymology.

In vivo incorporation fidelity has been traditionally assessed using reporter assays, mass spectrometry, or functional readouts, but these methods do not actively eliminate erroneous products. In *Escherichia coli*, the N-end rule pathway provides a native mechanism for targeted protein degradation based on the identity of the N-terminal amino acid. Certain residues—such as phenylalanine, leucine, tryptophan, and tyrosine—are recognized as “destabilizing” by the adaptor protein ClpS, which delivers substrates to the ClpAP protease complex for degradation. Conversely, other residues are “stabilizing” and evade degradation. However, this system has not been systematically leveraged to improve the selectivity of NSAA incorporation.

Current methods for evaluating or improving OTS fidelity are largely passive and lack real-time quality control. There remains a critical need for active, post-translational proofreading strategies that can distinguish between desired NSAAs and undesired amino acids—including standard amino acids and off-target NSAAs—and selectively degrade erroneous polypeptides. Such a system would not only enhance the accuracy of NSAA incorporation but also enable high-throughput evolution of more selective OTSs and improve the safety and efficiency of synthetic biology applications.

## SUMMARY

The present invention provides a method for selectively degrading polypeptides that contain undesired amino acids at a defined position, thereby enriching for target polypeptides that incorporate a desired non-standard amino acid (NSAA). This method leverages a post-translational proofreading (PTP) strategy based on the N-end rule pathway of protein degradation. In one embodiment, a target polypeptide is expressed as a fusion protein with a removable protecting group positioned immediately upstream of the amino acid incorporation site. Upon removal of this protecting group, the N-terminal residue of the mature polypeptide is exposed. If this residue is a destabilizing amino acid—either a standard amino acid or an undesired NSAA—the polypeptide is recognized by an adapter protein (e.g., ClpS in *E. coli*) and delivered to a protease (e.g., ClpAP) for degradation. If the residue is a stabilizing NSAA, the polypeptide escapes degradation and accumulates.

The invention further provides a method for making a target polypeptide with a desired NSAA at a specific site by expressing the polypeptide in a cell that has been genetically engineered to include: (1) a nucleic acid encoding the target polypeptide with a UAG (or other reassigned) codon at the target site; (2) an orthogonal aaRS/tRNA pair specific for the desired NSAA; (3) a removable protecting group (e.g., ubiquitin) fused to the N-terminus of the target polypeptide; and (4) an enzyme (e.g., a ubiquitin cleavase such as UBP1) that removes the protecting group post-translationally to expose the N-terminal residue. The cell may also express an adapter protein (e.g., ClpS or an engineered variant thereof) and a protease to implement the N-end rule degradation pathway.

In a preferred embodiment, the removable protecting group is ubiquitin, which is cleaved by a heterologous ubiquitin-specific protease (e.g., N-terminally truncated yeast UBP1) to expose the residue encoded by the UAG codon as the new N-terminus. The identity of this residue then determines the fate of the protein: destabilizing residues trigger degradation, while stabilizing residues permit accumulation. The invention includes engineered variants of ClpS with altered binding specificities that can be tuned to recognize or ignore specific NSAAs, thereby customizing the proofreading stringency.

The method enables active discrimination between desired and undesired amino acid incorporation events. By coupling NSAA incorporation to protein stability, the system provides a powerful selection pressure that can be used to evolve more selective orthogonal translation systems. For example, libraries of aaRS or tRNA variants can be screened in a host strain equipped with PTP machinery, where only cells producing the correct NSAA-incorporated protein will fluoresce (if a reporter like sfGFP is used) or survive (in biocontainment contexts).

The invention further encompasses iterative optimization of the production process by altering reaction conditions, engineering the aaRS or tRNA components, or modifying the adapter protein to refine recognition profiles. This cycle of expression, proofreading, and selection can be repeated until the desired level of fidelity and yield is achieved. The result is a robust platform for high-fidelity production of NSAA-containing proteins with applications in industrial biotechnology, therapeutic protein engineering, and synthetic biocontainment.

## DETAILED DESCRIPTION

The terms “polypeptide” and “protein” are used interchangeably herein to refer to a polymer of amino acid residues linked by peptide bonds. A “target polypeptide” is a polypeptide of interest that is engineered to contain a non-standard amino acid (NSAA) at a specific location. The invention involves the substitution of a standard amino acid with a NSAA at a defined position within the target polypeptide, typically via suppression of a reassigned codon such as UAG.

To implement the post-translational proofreading system, a host cell is genetically modified to express the target polypeptide as a fusion with a removable protecting group. This protecting group is positioned adjacent to the amino acid target location such that, upon removal, the residue at the target site becomes the N-terminal amino acid of the mature polypeptide. The cell is also engineered to express a protease system capable of degrading polypeptides based on the identity of their N-terminal residue. In prokaryotes such as *E. coli*, this system comprises the adapter protein ClpS and the ClpAP protease complex. In eukaryotes, analogous systems involving ubiquitin ligases and the proteasome may be employed.

The removable protecting group may be a peptide sequence produced by the cell, such as ubiquitin, or a foreign protein domain. In prokaryotic cells, ubiquitin—which is not natively present—can be used as a heterologous protecting group that is efficiently cleaved by a co-expressed ubiquitin cleavase, such as the N-terminally truncated yeast UBP1. In eukaryotic cells, however, endogenous deubiquitinating enzymes may interfere with controlled cleavage, necessitating the use of alternative cleavable linkers such as the MENLYFQ/* sequence, which is recognized by tobacco etch virus (TEV) protease. Self-splicing intein domains or methionine aminopeptidase-sensitive sequences may also serve as removable protecting groups.

The protease system is often placed under the control of an inducible promoter to allow temporal regulation of proofreading activity. Upregulation of the adapter protein (e.g., ClpS) enhances the degradation of polypeptides bearing N-terminal destabilizing residues. The N-end rule pathway is thus repurposed as a quality control mechanism: only polypeptides with N-terminal residues that are not recognized as destabilizing by the adapter protein will accumulate.

To facilitate detection and quantification, a detectable moiety is attached to the C-terminus of the target polypeptide. This may be a fluorescent protein such as superfolder green fluorescent protein (sfGFP), an epitope tag (e.g., His6x, FLAG, HA), or a reporter enzyme (e.g., luciferase). The presence and abundance of the target polypeptide can then be monitored by fluorescence, immunoblotting, or enzymatic activity.

The host cell is further engineered to include foreign genetic material encoding an orthogonal aaRS/tRNA pair that is cognate to the desired NSAA. This pair must be orthogonal—i.e., non-cross-reactive with endogenous synthetases and tRNAs—and capable of efficiently charging the tRNA with the NSAA and incorporating it at the reassigned codon. Exemplary synthetase families for bacterial hosts include those derived from *Methanocaldococcus jannaschii* TyrRS, which have been extensively engineered for NSAA incorporation.

Genomic recoding of the host cell—such as the deletion of all UAG stop codons and the release factor 1 (RF1)—enhances the efficiency and orthogonality of UAG suppression. Nucleic acids encoding the target polypeptide, the orthogonal translation system, the protecting group, the cleaving enzyme, and the adapter-protease system are introduced into the cell via vectors such as plasmids or integrated into the chromosome using recombineering techniques. Vectors may include regulatory elements such as inducible promoters (e.g., TET, arabinose), ribosome binding sites, and terminator sequences to ensure proper expression.

The invention is applicable to a wide range of host cells, including prokaryotes such as *Escherichia coli*, *Bacillus subtilis*, and other useful bacteria, as well as eukaryotes such as *Saccharomyces cerevisiae*, other fungal cells, plant cells, and mammalian cells. The choice of host depends on the application, scalability, and compatibility with the orthogonal system.

Standard amino acids are defined as the twenty naturally occurring proteinogenic amino acids. In *E. coli*, the N-end rule classifies Phe, Leu, Trp, and Tyr as primary destabilizing residues. The invention extends this classification by engineering the N-end rule to recognize additional residues—such as Ile, Val, Asp, Glu, Asn, and Gln—as destabilizing through overexpression of ClpS or introduction of auxiliary enzymes like aminoacyl-transferases.

Non-standard amino acids (NSAAs) are unnatural amino acids that are not among the twenty standard residues. Examples include p-acetylphenylalanine, p-azidophenylalanine, biphenylalanine (BipA), and 5-hydroxytryptophan. NSAAs may be functionalized with chemical handles (e.g., azides, alkynes, ketones) for bioorthogonal chemistry. The invention enables the classification of NSAAs as either N-end stabilizing or destabilizing based on their interaction with the adapter protein, and this classification can be altered by engineering the adapter protein’s binding pocket.

### II. Removable Protecting Groups

A removable protecting group is positioned immediately upstream of the amino acid target location in the nascent polypeptide chain. This group masks the N-terminal identity of the target residue until it is cleaved post-translationally. In one embodiment, the protecting group is a peptide sequence produced by the cell, such as ubiquitin, which is not naturally processed in prokaryotes but can be cleaved by a heterologous ubiquitin-specific protease. In eukaryotic cells, where ubiquitin is endogenous, alternative cleavable linkers such as the TEV protease recognition sequence (ENLYFQ/G) are preferred. The asterisk denotes the cleavage site, yielding the target residue as the new N-terminus.

Foreign removable protecting groups, such as engineered protein domains or synthetic peptides, may also be used. Enzyme-cleavable protecting groups include those recognized by TEV protease, factor Xa, or thrombin. Self-splicing intein domains can serve as autocatalytic protecting groups that excise themselves without exogenous enzymes. Methionine aminopeptidases, which remove N-terminal methionine when followed by small residues, can also be exploited to expose the target residue if the protecting group is designed accordingly.

### III. Detectable Moiety

A detectable moiety is fused to the C-terminus of the target polypeptide to enable monitoring of expression and stability. This may be a fluorescent protein such as sfGFP, which provides a quantitative readout via fluorescence intensity. Alternatively, epitope tags (e.g., His6x, FLAG) allow detection by immunoassays, while reporter genes (e.g., lacZ, luc) enable enzymatic quantification. The choice of detectable moiety depends on the screening or analytical method employed.

### IV. Genetic Modifications

The host cell is genetically modified to include nucleic acid sequences encoding the target polypeptide, the orthogonal translation system, the protecting group, the cleaving enzyme, and the degradation machinery. The target polypeptide is encoded with a reassigned codon (e.g., UAG) at the target site. The removable protecting group and detectable moiety are fused in-frame. The cell is genomically recoded to remove competing stop codons and release factors. Foreign nucleic acids encoding the aaRS, tRNA, ubiquitin cleavase, ClpS, and ClpP are introduced via plasmids or chromosomal integration using methods such as λ Red recombineering, MAGE, or Gibson assembly. Vectors include regulatory elements like inducible promoters and terminators.

### V. Adapter Protein Protease Systems

The protease system includes an adapter protein (e.g., ClpS) that recognizes N-terminal destabilizing residues and delivers substrates to a protease (e.g., ClpAP). In *E. coli*, ClpS binds Phe, Leu, Trp, Tyr, and certain NSAAs, targeting them for degradation. Engineered ClpS variants with altered specificities can be used to customize proofreading stringency.

### VI. Cells

The invention is practiced in prokaryotic or eukaryotic cells. Prokaryotic hosts include *E. coli*, *B. subtilis*, and other bacteria. Eukaryotic hosts include *S. cerevisiae*, other fungi, plant cells, and mammalian cells. The *E. coli* strain C321.ΔA, which lacks UAG codons and RF1, is a preferred host for UAG suppression.

### VII. Standard Amino Acid

Standard amino acids include the twenty natural residues. In *E. coli*, the N-end rule destabilizing residues are Phe, Leu, Trp, Tyr. Overexpression of ClpS can extend degradation to Ile, Val, and others. Engineering the pathway can convert Asp, Glu, Asn, Gln into destabilizing residues via aminoacylation.

### VIII. Non-Standard Amino Acid

NSAAs are unnatural amino acids, such as BipA, pAcF, pAzF. They are incorporated via orthogonal aaRS/tRNA pairs. NSAAs may be classified as N-end stabilizing or destabilizing based on ClpS binding. Engineered ClpS variants can alter this classification.

## Exemplary Degradation Materials and Methods

A protein production method involves providing a genomically recoded *E. coli* cell with a nucleic acid encoding Ub-UAG-sfGFP, a plasmid expressing BipARS and tRNA^Tyr_CUA, and the NSAA BipA. Co-expression of UBP1 cleaves ubiquitin, exposing the N-terminal residue. If BipA is incorporated, sfGFP is stable; if Tyr/Leu is misincorporated, ClpS targets it for degradation. GFP fluorescence approximates NSAA incorporation fidelity. Screening synthetase variants in a ClpS-overexpressing strain enables isolation of high-fidelity OTSs.

## Example II

Genetic code expansion using OTSs suffers from promiscuity. Post-translational proofreading (PTP) addresses this by coupling NSAA identity to protein stability via the N-end rule. A Ub-UAG-sfGFP reporter was integrated into *E. coli* C321.ΔA. Expression of UBP1 exposed the UAG-encoded residue as the N-terminus. Without BipA, misincorporation of Tyr/Leu led to degradation upon ClpS overexpression, reducing GFP signal. With BipA, signal remained high, confirming BipA is N-end stabilizing. PTP was extended to other NSAAs: large hydrophobic NSAAs (e.g., p-iodo-Phe) were stabilizing, while smaller ones (e.g., p-azido-Phe) were destabilizing. ClpS was engineered to alter NSAA recognition: ClpS_V65I stabilized all tested phenyl-NSAAs while maintaining degradation of standard AAs. This enabled high-fidelity OTS evolution.

## Example III

ClpS variants were rationally engineered by mutating hydrophobic residues in the binding pocket (positions 32, 43, 65, 99). ClpS_V65I and ClpS_L32F exhibited tunable specificities, distinguishing between standard AAs and NSAAs with high precision. These variants enabled customized proofreading for different NSAA classes.

## Example IV

Local sense codon reassignment was achieved by combining PTP with OTSs that suppress sense codons, allowing incorporation of NSAAs at non-stop positions while maintaining fidelity through post-translational quality control.

## Example V

The *E. coli* strain C321.ΔA was modified via λ Red recombineering and MAGE to integrate a TET-promoter-driven Ub-UAG-sfGFP cassette and delete mutS and clpS. Plasmids expressing UBP1 and ClpS_V65I were integrated to create the C321.Nend strain. OTS libraries were generated by error-prone PCR of bipARS and screened via FACS: positive sort in BipA+, negative sort in BipA−, then final positive sort. Variant 10 showed high selectivity for BipA over other NSAAs and standard AAs. Mass spectrometry confirmed site-specific incorporation. tRNA mutations (e.g., G51A) enhanced selectivity by modulating EF-Tu binding.

## Example VI

A method of making a target polypeptide involves genetically modifying a cell to express Ub-target-sfGFP, an orthogonal aaRS/tRNA pair, UBP1, and ClpS_V65I. The cell is provided with the NSAA. Upon expression, UBP1 cleaves ubiquitin, exposing the N-terminal residue. If it is a standard AA or undesired NSAA, ClpS_V65I targets it for degradation. If it is the desired NSAA, the polypeptide accumulates. This enriches for correct incorporation and enables high-yield, high-fidelity production of NSAA-containing proteins.