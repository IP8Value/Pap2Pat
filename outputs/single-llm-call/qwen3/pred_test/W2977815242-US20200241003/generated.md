# DESCRIPTION

## BACKGROUND

- T-cell epitopes are short peptide fragments derived from intracellular proteins that are presented on the surface of antigen-presenting cells by major histocompatibility complex class I molecules, enabling recognition by cytotoxic T lymphocytes. These interactions are central to adaptive immune responses against pathogens, tumors, and self-antigens, yet the identification of immunogenic epitopes remains a formidable challenge due to the vast combinatorial diversity of potential peptide sequences and the polymorphic nature of human leukocyte antigen (HLA) molecules that govern peptide binding. The human genome encodes hundreds of HLA class I alleles, each with distinct peptide-binding motifs, and the probability of any given 8–12 amino acid sequence being both processed by the proteasome, transported into the endoplasmic reticulum, loaded onto an HLA molecule, and subsequently recognized by a cognate T-cell receptor is exceedingly low. This complexity is compounded by the fact that T-cell recognition is not solely determined by peptide–MHC affinity but is also influenced by the kinetics of TCR engagement, co-receptor signaling, intracellular antigen abundance, and the efficiency of antigen processing. Conventional methods for epitope discovery rely on labor-intensive, low-throughput assays in which candidate peptides are individually synthesized, loaded onto target cells, and tested for T-cell activation through cytokine secretion or cytotoxicity readouts. While pooling strategies have been employed to increase screening capacity, they necessitate iterative deconvolution steps that are time-consuming, prone to false negatives due to epitope competition, and incapable of resolving epitopes present at low frequencies within complex mixtures. As a result, exhaustive interrogation of proteome-wide epitope spaces remains impractical, limiting the ability to comprehensively map T-cell reactivity in clinical contexts such as cancer immunotherapy, autoimmune disease, and vaccine development.

- Major histocompatibility complex class I molecules are heterotrimeric complexes composed of a polymorphic heavy chain, β2-microglobulin, and a bound peptide ligand. These molecules are expressed on nearly all nucleated cells and are responsible for presenting endogenously synthesized peptides to CD8+ cytotoxic T lymphocytes. The peptide-binding groove of MHC class I is highly selective, accommodating peptides of defined length, typically 8 to 12 amino acids, with anchor residues that conform to allele-specific binding motifs. The genetic diversity of MHC class I loci—particularly within the HLA-A, HLA-B, and HLA-C genes in humans—results in an immense population-level variation in peptide presentation capacity, making it difficult to develop universal epitope prediction tools or broadly applicable screening platforms. Furthermore, the intracellular machinery responsible for generating MHC class I ligands—including the proteasome, immunoproteasome, endoplasmic reticulum aminopeptidases, and the transporter associated with antigen processing—introduces additional layers of bias and stochasticity that cannot be fully recapitulated by synthetic peptide loading. Consequently, epitopes identified by affinity-based methods using recombinant pMHC complexes often fail to correlate with immunogenicity in physiological settings, as they bypass the natural pathways of antigen processing and presentation that are essential for authentic T-cell activation.

- The discovery of T-cell epitopes is hindered by multiple interrelated technical and biological constraints. First, the theoretical space of possible 9-mer peptides exceeds 10^12 unique sequences, rendering exhaustive screening infeasible without intelligent sampling or enrichment strategies. Second, most functional assays require the presentation of individual peptides on the surface of target cells, which demands either chemical synthesis of hundreds to thousands of peptides or transfection of individual expression constructs, neither of which scales efficiently. Third, the sensitivity of conventional assays is limited by the requirement for high-frequency T-cell populations, as low-abundance antigen-specific clones are often masked by polyclonal background reactivity. Fourth, many methods rely on surrogate readouts such as NFAT-driven fluorescence or cytokine release, which do not directly reflect the cytotoxic effector function that is most relevant to immune surveillance and therapeutic efficacy. Finally, the transient and low-affinity nature of TCR–pMHC interactions means that epitopes with suboptimal binding but high functional relevance may be overlooked by methods that prioritize binding strength over biological activity. These limitations collectively impede the systematic identification of immunogenic epitopes from complex biological samples, particularly in the context of patient-derived tissues where antigenic targets are unknown and T-cell repertoires are heterogeneous.

- There exists a critical and unmet need for a high-throughput, function-based method capable of simultaneously interrogating vast libraries of candidate epitopes while preserving the physiological context of antigen processing, presentation, and T-cell recognition. Current approaches either sacrifice biological relevance for scalability—such as pMHC multimer staining or single-chain MHC display—or maintain physiological fidelity at the cost of throughput—such as peptide pulsing and ELISpot. A method that integrates the scalability of next-generation sequencing with the functional specificity of cytotoxic T-cell killing would enable the unbiased discovery of epitopes from entire proteomes, tumor transcriptomes, or pathogen databases without prior assumptions about immunogenicity. Such a method would be transformative for personalized cancer immunotherapy, where the identification of patient-specific neoantigens is essential for designing effective vaccines or adoptive T-cell therapies. It would also enhance the development of vaccines for emerging infectious diseases, improve the prediction of alloreactive epitopes in transplantation, and facilitate the discovery of autoantigens in autoimmune disorders. The absence of a method that meets these criteria has constrained progress in understanding the true breadth of T-cell reactivity and its role in health and disease.

## SUMMARY OF THE INVENTION

- The invention provides a novel method for identifying T-cell epitopes through the co-expression of candidate epitope-encoding minigenes and a granzyme B-sensitive fluorescent reporter system within antigen-presenting cells, enabling the direct isolation of target cells that have been recognized and attacked by cytotoxic T lymphocytes. This approach leverages the natural biological pathway of cytotoxic T-cell-mediated killing to selectively recover antigen-presenting cells bearing immunogenic peptides, thereby bypassing the need for synthetic peptide loading, cytokine detection, or artificial signaling reporters. The method enables the simultaneous screening of millions of candidate epitopes in a single experiment, with high sensitivity and specificity, and without requiring prior knowledge of the antigenic sequence.

- The method employs genetically engineered reporter cells that stably express a fusion protein consisting of two fluorescent domains, cyan fluorescent protein and yellow fluorescent protein, separated by a peptide linker that is specifically cleaved by granzyme B. Upon recognition of a cognate peptide–MHC complex by a cytotoxic T cell, granzyme B is delivered into the target cell via the perforin-dependent pathway, resulting in cleavage of the reporter protein and a measurable shift in fluorescence emission from Förster resonance energy transfer (FRET) to free cyan fluorescent protein signal. This shift is detectable by flow cytometry and allows for the physical separation of target cells that have been engaged by antigen-specific T cells from those that remain untargeted.

- The steps of the method include: (1) constructing a library of nucleic acid sequences encoding candidate epitopes as minigenes fused to the granzyme B-cleavable FRET reporter; (2) transducing a population of antigen-presenting cells with the library at a low multiplicity of infection to ensure single minigene integration per cell; (3) co-culturing the transduced cells with a population of cytotoxic T cells of interest; (4) isolating cells exhibiting a FRET-shift signal by fluorescence-activated cell sorting; and (5) recovering and sequencing the integrated minigenes from the sorted cells to identify the epitopes recognized by the T-cell population. The method is scalable, sensitive, and applicable to both monoclonal and polyclonal T-cell populations.

- Reporter cells are isolated by fluorescence-activated cell sorting based on the loss of FRET signal and the concomitant increase in cyan fluorescent protein emission, which occurs only in cells that have received granzyme B delivery from activated cytotoxic T cells. This sorting step enables the physical separation of antigen-presenting cells that have been targeted by T cells from those that have not, thereby enriching for cells bearing immunogenic epitopes without the need for additional labeling or reporter gene activation in the T cells themselves.

- The FRET-based fluorescent protein signaling system comprises a fusion protein in which cyan fluorescent protein and yellow fluorescent protein are linked by a peptide sequence that serves as a substrate for granzyme B. In the uncleaved state, excitation of cyan fluorescent protein results in energy transfer to yellow fluorescent protein, producing a FRET signal. Upon cleavage by granzyme B, the two fluorophores are separated, abolishing FRET and restoring the intrinsic emission of cyan fluorescent protein. This change in spectral signature is robust, quantifiable, and compatible with standard flow cytometry instrumentation.

- An alternative method for determining epitopes involves the use of a leuco-dye-based system in which a cell-permeable, non-fluorescent dye is converted into a fluorescent product upon cleavage by granzyme B, enabling detection by microscopy or flow cytometry. This system provides a non-genetic alternative for epitope detection in primary cells or cell lines that are refractory to genetic modification.

- The invention further comprises an enriched library of candidate epitope-encoding nucleic acids, wherein each member of the library encodes a peptide sequence of 8 to 40 amino acids that is capable of being processed and presented by major histocompatibility complex class I molecules. The library may be derived from genomic DNA, complementary DNA, synthetic oligonucleotides, or overlapping peptide fragments of proteins of interest, and may include degenerate codons to maximize sequence diversity while minimizing redundancy.

- The method is used to test for cellular immunity by exposing reporter cells, bearing a library of candidate epitopes, to a sample of T cells derived from a patient or experimental subject. The presence of a FRET-shift signal following co-culture indicates that the T-cell population contains clones capable of recognizing one or more of the encoded epitopes, thereby providing a direct readout of antigen-specific cellular immunity.

- The method is employed to identify epitopes for use in vaccination by screening T-cell populations from vaccinated individuals against libraries of pathogen-derived or tumor-associated antigens, thereby revealing the immunodominant epitopes that drive protective immune responses. The identified epitopes can then be incorporated into synthetic vaccines or used to monitor immune responses in clinical trials.

- The method is adapted to identify epitopes associated with autoimmune disorders by screening T cells isolated from affected tissues—such as pancreatic islets in type 1 diabetes or synovial fluid in rheumatoid arthritis—against libraries of self-proteins. Epitopes that are preferentially recognized by autoreactive T cells can be targeted for tolerance induction or used as biomarkers for disease progression.

- The method is further used to identify epitopes that induce immune tolerance by screening T cells from regulatory or anergic populations against libraries of self-antigens. Epitopes that elicit a lack of FRET-shift signal in the presence of regulatory T cells may represent candidates for therapeutic tolerance induction.

- The method enables the identification of epitopes recognized by public T-cell clonotypes—T-cell receptors that are shared across multiple individuals—by screening T-cell repertoires from unrelated donors against the same epitope library. Epitopes consistently recognized across multiple individuals are likely to be immunodominant and suitable for broad-coverage vaccine design.

- The invention includes the expression of encoded DNA, RNA, or peptide libraries within antigen-presenting cells via lentiviral transduction, plasmid transfection, or other gene delivery systems. The libraries are designed to encode peptides of defined length that are flanked by sequences that facilitate proteasomal processing and MHC class I binding, ensuring physiological relevance of epitope presentation.

- Detection and isolation of reporter cells are achieved by flow cytometry based on the FRET-shift signature, which distinguishes cells that have been targeted by cytotoxic T cells from bystander cells. The sorted cells are then subjected to nucleic acid extraction and amplification to recover the integrated minigenes responsible for epitope expression.

- Analysis of the recovered cells involves deep sequencing of the amplified minigenes, followed by bioinformatic filtering to identify sequences that are significantly enriched in the FRET-shifted population relative to the unshifted control. The identified sequences are then validated as epitopes through independent functional assays.

## DETAILED DESCRIPTION OF THE INVENTION

- Conventional techniques for T-cell epitope discovery include peptide pulsing of dendritic cells followed by cytokine ELISpot or intracellular staining, MHC tetramer staining, and computational prediction algorithms based on binding affinity. These methods are limited by low throughput, high false-negative rates, and an inability to detect epitopes presented at low abundance or recognized by low-frequency T-cell clones. None of these approaches enable the unbiased, high-throughput screening of entire proteomes or complex antigen libraries in a physiologically relevant context.

- Reporter cells used in the invention are genetically modified to express both a candidate epitope-encoding minigene and a granzyme B-cleavable FRET reporter protein. These cells serve as surrogate antigen-presenting cells that display endogenously processed peptides on MHC class I molecules and simultaneously report T-cell recognition through a fluorescent signal change. The reporter cells are engineered to maintain normal antigen processing machinery, ensuring that epitope presentation occurs through the same pathways as in vivo.

- The use of MHC-matched reporter cells is essential to ensure that the peptides encoded by the minigene library are properly loaded onto MHC class I molecules and presented in a manner that is recognizable by the T-cell population under investigation. Mismatched MHC alleles prevent epitope presentation and result in false-negative outcomes. The invention utilizes reporter cells derived from the same species and HLA haplotype as the T-cell source to preserve physiological relevance.

- The detectable signal generated by the invention is a shift in fluorescence emission from FRET to free cyan fluorescent protein upon cleavage of the reporter fusion protein by granzyme B. This signal is intrinsic to the target cell and does not require exogenous dyes, reporter gene induction, or T-cell labeling, making it highly specific and minimally perturbing to the biological system.

- Reporter cell recognition is illustrated by the co-culture of transduced cells with cytotoxic T cells, wherein only those target cells expressing a peptide that matches the T-cell receptor specificity are killed, resulting in granzyme B delivery and FRET signal loss. The remaining cells, which do not express a recognizable epitope, retain their FRET signal and serve as internal controls.

- The effector response elicited by cytotoxic T cells involves the polarization of lytic granules toward the immunological synapse, followed by the release of perforin and granzyme B into the target cell cytoplasm. Perforin facilitates the entry of granzyme B, which then cleaves the FRET reporter protein, initiating the detectable signal change. This process is highly specific and does not occur in bystander cells.

- Signal generation is illustrated by the excitation of cyan fluorescent protein at 405 nm, which, in the uncleaved state, produces emission at 525 nm due to FRET to yellow fluorescent protein. Upon cleavage, the emission shifts to 470 nm, corresponding to the free cyan fluorescent protein. This spectral shift is quantified using flow cytometry and used to gate and isolate targeted cells.

- The method for determining epitopes involves the introduction of a diverse library of minigenes into reporter cells, followed by co-culture with a T-cell population, sorting of FRET-shifted cells, and sequencing of the integrated minigenes to identify the epitopes responsible for T-cell recognition. The entire process is performed in a single round without the need for iterative panning or deconvolution.

- The steps of the method include: (1) cloning a library of candidate epitope-encoding sequences into a lentiviral vector downstream of a granzyme B-cleavable FRET reporter; (2) producing lentiviral particles and transducing antigen-presenting cells at a low MOI to ensure single-copy integration; (3) expanding transduced cells to generate a heterogeneous population; (4) co-culturing with expanded cytotoxic T cells; (5) performing fluorescence-activated cell sorting to isolate cells exhibiting FRET-shift; (6) extracting genomic DNA from sorted cells; (7) amplifying the integrated minigenes by PCR; and (8) sequencing the amplicons by next-generation sequencing to identify enriched epitope sequences.

- Analysis of epitope-encoding nucleic acids is performed by aligning sequencing reads to a reference library, calculating the relative abundance of each sequence in the FRET-shifted versus unshifted populations, and applying statistical thresholds to identify sequences significantly enriched in the targeted fraction. Sequences that exceed a defined fold-enrichment and statistical significance cutoff are considered candidate epitopes.

- High-throughput epitope screening is illustrated by the simultaneous testing of over one million unique minigene sequences in a single experiment, with detection of epitopes present at frequencies as low as one in ten thousand. This level of sensitivity far exceeds that of conventional methods and enables the discovery of rare or subdominant epitopes.

- Genetic modification of reporter cells is achieved by stable transduction with lentiviral vectors encoding the FRET reporter and minigene library. The cells are engineered to express endogenous MHC class I molecules and antigen processing machinery, ensuring physiological relevance. The reporter construct is designed to be silent until a minigene is inserted, minimizing background fluorescence.

- Schemes for high-throughput epitope screening involve the use of multi-well plate formats, pooled transductions, and automated cell sorting to enable parallel screening of multiple T-cell populations against the same library. The system is scalable to accommodate large clinical cohorts and diverse antigen sources.

- Bioinformatics methods are employed to filter sequencing data, remove PCR and sequencing artifacts, cluster highly similar sequences, and predict MHC binding affinity using algorithms such as NetMHCpan. These methods enhance the accuracy of epitope identification and reduce false positives.

- Iterative determination of T-cell epitopes is performed by using the top enriched sequences from an initial screen to construct a focused secondary library, which is then re-screened against the same T-cell population to confirm immunogenicity and eliminate background noise.

- The enriched library of candidate epitope-encoding nucleic acids is constructed by synthesizing degenerate oligonucleotides encoding random or biased peptide sequences, ligating them into a lentiviral vector downstream of the FRET reporter, and transducing target cells to generate a diverse population of reporter cells. The library may contain between 10^5 and 10^7 unique members.

- Exemplary frequencies of epitope detection demonstrate that epitopes present at a frequency of 1 in 10,000 can be reliably identified with a signal-to-noise ratio exceeding 10 standard deviations above background, even in the presence of polyclonal T-cell populations.

### Iterative Determination of T Cell Epitopes

- The iterative method involves conducting an initial unbiased screen of a large, diverse minigene library to identify candidate epitopes that are enriched in the FRET-shifted population. These candidates are then synthesized as a focused oligonucleotide pool and re-cloned into the reporter vector to generate a secondary library with reduced complexity but higher representation of top candidates.

- The steps of the iterative method include: (1) performing a primary screen of a random minigene library against a T-cell population; (2) sequencing and ranking enriched sequences based on fold-enrichment and statistical significance; (3) synthesizing a subset of the top-ranked sequences as a custom oligonucleotide pool; (4) cloning the pool into the reporter vector to generate a secondary library; (5) transducing reporter cells with the secondary library; (6) co-culturing with the same T-cell population; and (7) sequencing the recovered minigenes to confirm enrichment and eliminate false positives.

- The identifying step involves the statistical comparison of sequence abundance between FRET-shifted and unshifted populations to determine which minigenes are significantly enriched. A threshold of 10 standard deviations above the mean Δ relative abundance is used to define a positive hit.

- Repeating cycles of screening and refinement allows for the progressive enrichment of true epitopes while filtering out non-specific or background sequences. After two rounds of iteration, the majority of false positives are eliminated, and the true epitope is confirmed with high confidence.

- Exemplary frequencies demonstrate that in a secondary screen, a previously identified epitope such as SIINFEKL or KVPRNQDWL is recovered at frequencies exceeding 100-fold enrichment over background, while non-specific sequences that were enriched in the primary screen are no longer detected.

### Reporter Cells

- Reporter cells are capable of endogenous antigen processing, MHC class I presentation, and susceptibility to granzyme B-mediated apoptosis, enabling them to faithfully recapitulate the biological conditions of T-cell recognition in vivo. They are engineered to express the FRET reporter constitutively and to maintain normal levels of proteasomal and transporter activity.

- The MHC class I molecule is essential for the presentation of the encoded epitopes to CD8+ T cells. The reporter cells are selected or engineered to express MHC alleles that match those of the T-cell source to ensure physiological compatibility and epitope presentation.

- Autologous cells, such as patient-derived dendritic cells or B lymphoblastoid cell lines, may be used as reporter cells to preserve individual-specific HLA haplotypes and antigen processing profiles, enhancing the clinical relevance of the assay.

- An immortalized antigen-presenting cell line, such as EL4 or ID8, is used as a standardized platform for high-throughput screening. These cell lines are genetically stable, easily transducible, and express high levels of MHC class I, making them ideal for reproducible experiments.

- Transfection and transduction are used to introduce the reporter construct into target cells. Lentiviral transduction is preferred due to its high efficiency, stable integration, and ability to transduce both dividing and non-dividing cells.

- A viral vector, specifically a lentiviral vector, is used to deliver the reporter construct and minigene library into target cells. The vector is pseudotyped with VSV-G to enable broad tropism and high titer production.

- The signal generating product is a fusion protein comprising cyan fluorescent protein, a granzyme B cleavage site, and yellow fluorescent protein. Cleavage of this protein by granzyme B results in loss of FRET and gain of cyan fluorescence.

- The FRET-based signaling system is characterized by a high signal-to-noise ratio, rapid kinetics, and compatibility with standard flow cytometry instruments. The system responds within hours of T-cell engagement and is stable enough to permit cell sorting.

- The leuco-dye-based system employs a cell-permeable, non-fluorescent substrate that is cleaved by granzyme B to produce a fluorescent product, offering an alternative to genetic reporters in primary cells.

- Exemplary signaling systems include the CFP-YFP FRET pair and a leuco-dye derivative such as Ac-DEVD-AMC, which emits fluorescence upon cleavage by granzyme B.

- The genetic construct consists of a lentiviral backbone containing a promoter, the FRET reporter cassette with a stuffer sequence upstream of the minigene insertion site, and flanking restriction sites for cloning. Upon minigene insertion, the stuffer is replaced, enabling expression of the full reporter.

- The FRET-based signaling system is illustrated by the excitation of CFP at 405 nm, which, in the uncleaved state, produces emission at 525 nm due to energy transfer to YFP. Upon cleavage, emission shifts to 470 nm, corresponding to free CFP.

- Protein expression is driven by a constitutive promoter such as EF1α or PGK, ensuring uniform reporter expression across the cell population. The minigene is inserted in-frame between the FRET domains to ensure proper folding and cleavage.

- Exemplary constructs include pMND-silent-FRET, which contains a silent stuffer sequence that is replaced upon minigene insertion, and pCCL-c-MNDU3-PGK-FRET, which contains a minimal promoter and optimized polyadenylation signal.

- The lentivirus construct is produced by co-transfecting HEK293T cells with the transfer plasmid, packaging plasmid (pCMV-ΔR8.91), and envelope plasmid (pCMV-VSV-G). Supernatants are harvested, concentrated by ultracentrifugation, and titrated by transduction of HeLa cells.

- Analysis of epitope-encoding nucleic acids is performed by extracting genomic DNA from sorted cells, amplifying the integrated minigenes using primers flanking the insertion site, and sequencing the amplicons on an Illumina MiSeq platform.

- Exemplary human cell lines include K562 cells transduced to express HLA-A*02:01, HLA-B*07:02, or other common HLA alleles, enabling screening of human T-cell populations in a controlled, MHC-matched environment.

### Epitope-Encoding Nucleic Acid Libraries

- Epitope-encoding nucleic acid libraries are collections of DNA sequences that encode peptides of varying lengths, typically between 8 and 40 amino acids, designed to be processed and presented by MHC class I molecules. The libraries are constructed to maximize sequence diversity while maintaining physiological relevance.

- The size and structure of member sequences vary to include both minimal epitopes and extended flanking regions that influence proteasomal processing. Some members encode 9-mer epitopes, while others encode 16-mer or 40-mer sequences that contain multiple potential epitopes.

- The libraries encode peptides capable of being processed by the endogenous antigen presentation machinery, including sequences with proteasomal cleavage sites, endoplasmic reticulum signal peptides, and TAP transporter motifs.

- The libraries encode peptides of various lengths to account for differences in MHC binding preferences and processing efficiency. Shorter peptides may be optimal for certain HLA alleles, while longer peptides may better reflect natural antigen processing.

- Libraries are constructed by encoding peptides and polypeptides derived from full-length proteins, including overlapping fragments that span entire proteomes or pathogen genomes. This ensures comprehensive coverage of potential epitopes.

- Libraries are derived from protein sequences by generating overlapping peptide segments with defined offsets, such as 1- or 2-amino acid shifts, to ensure complete sampling of the sequence space.

- Codon substitutions are introduced using degenerate codons to maximize sequence diversity while minimizing redundancy. For example, a position encoding leucine may be represented by six different codons to increase coverage.

- Coding segments are synthesized using array-based oligonucleotide synthesis, allowing for the production of millions of unique sequences in a single reaction. Synthesized oligos are amplified by PCR and ligated into a lentiviral vector.

- Synthesized segments are ligated into a vector using restriction enzymes or Gibson assembly to generate a library of reporter constructs ready for transduction.

- The size of epitope-encoding libraries varies from 10^5 to 10^7 unique members, depending on the desired coverage and the complexity of the antigen source.

- Libraries are prepared from cDNA or gDNA extracted from an individual’s tumor, infected tissue, or peripheral blood mononuclear cells, enabling patient-specific epitope discovery.

- Libraries are derived from cancer antigen-discovery techniques such as mass spectrometry of eluted MHC peptides or RNA sequencing of tumor transcriptomes to focus on biologically relevant targets.

- Selection of epitope-encoding nucleic acids is guided by in silico T-cell epitope prediction methods, such as NetMHCpan or IEDB, to enrich for sequences with predicted MHC binding affinity.

- Protein-encoding nucleic acids are obtained by sequencing an individual’s genome or transcriptome to identify somatic mutations, viral integrations, or aberrantly expressed genes that may serve as sources of neoantigens.

### Cytotoxic T-Cells

- Cytotoxic T cells are obtained from various sources, including peripheral blood mononuclear cells, tumor-infiltrating lymphocytes, lymph nodes, and spleens of immunized or diseased individuals. The source is selected based on the biological context of interest.

- T cells are obtained from tissues affected by diseases such as cancer, chronic infection, or autoimmunity to identify epitopes relevant to the pathological state. Tumor tissue is a preferred source for cancer immunotherapy applications.

- T cells are expanded in vitro using stimulation with anti-CD3 and anti-CD28 antibodies in the presence of interleukin-2 to generate sufficient numbers for screening.

- Tumor-infiltrating lymphocytes are obtained from tumor specimens by mechanical and enzymatic dissociation, followed by filtration and magnetic or fluorescence-activated cell sorting to isolate CD8+ T cells.

- Tumor tissues are excised under sterile conditions, minced, and digested with collagenase and DNase to generate single-cell suspensions. The resulting cells are filtered and washed to remove debris.

- TIL microcultures are expanded by plating single cells in 96-well plates with feeder cells, irradiated autologous PBMCs, and cytokines to support clonal outgrowth.

- TILs are cultured from single-cell digests by limiting dilution in the presence of IL-2 and anti-CD3/CD28 stimulation to isolate individual T-cell clones.

- TILs are cultured from disaggregation of tumor tissue using enzymatic and mechanical methods to preserve viability and functionality.

- TIL activity and specificity are assayed by co-culture with target cells expressing candidate epitopes and measuring FRET-shift or cytokine secretion.

- TIL activity is determined by cytokine secretion, particularly interferon-gamma, using ELISA or intracellular staining after stimulation with autologous tumor cells or peptide-pulsed APCs.

- TILs are stimulated with autologous tumor cell lines to activate epitope-specific clones and enhance their frequency prior to screening.

- IFN-γ secretion is quantified by ELISA using capture and detection antibodies specific for human or murine interferon-gamma, with standard curves generated using recombinant cytokine.

- TILs are obtained from multiple original wells to ensure representation of diverse clonotypes and to avoid bias from clonal dominance.

- TIL cultures are maintained separately to preserve clonal diversity and to allow for parallel screening against different epitope libraries.

- TIL activity and specificity are assessed by FRET-shift assay, cytokine release, and tetramer staining to validate the functional relevance of identified epitopes.

### Nucleic Acid Sequencing Techniques

- Nucleotide sequences of epitope-encoding nucleic acids are determined using next-generation sequencing platforms, including Illumina MiSeq, NovaSeq, or Ion Torrent systems, to achieve high depth and accuracy.

- Commercial DNA sequence analyzers are used to generate paired-end reads of sufficient length to fully cover the minigene insert, typically 2 × 250 bp or longer.

- Sequencing is performed with indexed primers to allow multiplexing of multiple samples in a single run, reducing cost and increasing throughput.

- Raw sequencing data are processed using bioinformatic pipelines to assemble reads, remove adapter sequences, filter low-quality reads, and cluster near-identical sequences to correct for PCR and sequencing errors.

### Assessing Cellular Immunity to Specific Antigens

- The cellular immunity of an individual to specific antigens is tested by exposing reporter cells bearing a library of candidate epitopes to a sample of T cells isolated from the individual’s blood, tissue, or lymph node.

- The presence and/or level of cellular immunity is determined by the magnitude of FRET-shift signal following co-culture, with higher signal indicating a stronger or more frequent T-cell response to one or more encoded epitopes.

- The assay can be used to monitor immune responses over time, assess vaccine efficacy, or identify epitopes associated with disease progression or remission.

### Further Applications

- The method is used to identify T-cell–antigen interactions in a wide range of diseases, including cancer, viral infections, bacterial infections, fungal infections, and autoimmune disorders.

- The methods are applied in cancer vaccine design by identifying patient-specific neoantigens that elicit strong T-cell responses, which are then incorporated into personalized mRNA or peptide vaccines.

- The methods are used in autologous cell therapy to select T-cell clones that recognize tumor-specific epitopes, which are then expanded and reinfused into the patient.

- The methods improve tissue matching between donors and recipients in transplantation by identifying minor histocompatibility antigens that trigger alloreactive T-cell responses, enabling better donor selection and immunosuppression strategies.

### Kits

- Kits include mammalian reporter cells stably transduced with a granzyme B-cleavable FRET reporter and a stuffer sequence upstream of the minigene insertion site.

- Kits include vectors for transducing reporter cells, including lentiviral plasmids encoding the reporter construct, packaging elements, and VSV-G envelope.

- Kits include detailed instructions for transducing reporter cells, including protocols for virus production, transduction efficiency optimization, and cell expansion.

- Kits include vectors for transfecting and/or transducing reporter cells with custom epitope libraries, including pre-cloned libraries of tumor-associated antigens, viral proteomes, or synthetic minigenes.

## Example 1

### Model Assays Based on Stably Transfected Mouse Lymphoblastic Cell Lines EL4 and EG7 as Model Reporter Cells

- Cytotoxic T lymphocytes derived from OT-I and pmel-1 transgenic mice were co-cultured with EL4 and EG7 cell lines stably transfected with minigenes encoding the SIINFEKL or KVPRNQDWL epitopes, respectively. Upon recognition, the T cells delivered granzyme B into the target cells, resulting in cleavage of the FRET reporter and a measurable shift in fluorescence. The FRET-shift signal was detected within four hours and was specific to the cognate epitope, with no signal observed in control cells expressing scrambled sequences. Apoptosis of target cells was confirmed by propidium iodide staining, which occurred after peak FRET-shift signal, demonstrating that the reporter system detects early events in cytotoxic killing prior to cell death.

## Example 2

### Confirming Function of Granzyme B-Sensitive Signal Generation Product

- The granzyme B-sensitive signal generating product was tested by treating reporter cells with purified granzyme B in vitro. Cleavage of the FRET reporter resulted in a dose-dependent loss of FRET signal and recovery of cyan fluorescent protein emission, confirming that the signal is directly mediated by granzyme B and not by other proteases. Inhibition of granzyme B with specific inhibitors abolished the signal, while inhibitors of caspases or other apoptotic proteases had no effect, demonstrating the specificity of the system.

## Example 3

### Model Assays Based On Stably Transfected Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

- ID8 cells stably transfected with Ova minigenes were mixed with cells expressing scrambled minigenes at a 1:1 ratio and co-cultured with OT-I T cells. Following FACS sorting based on FRET-shift, genomic DNA was extracted and minigenes were amplified by PCR and sequenced. The Ova minigene was enriched by over 100-fold in the FRET-shifted population, while the scrambled minigene was equally represented in both shifted and unshifted gates. This demonstrated that the method can detect a single epitope within a complex mixture of irrelevant sequences.

## Example 4

### Model Assays Based on Lentivirus-Transduced Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

- ID8 cells were transduced with lentivirus encoding the Ova minigene and FRET reporter. After selection, cells were co-cultured with OT-I T cells and sorted based on FRET-shift. Minigenes recovered from FRET-shifted cells were sequenced and found to contain the Ova sequence with high frequency, while control cells transduced with empty vector showed no enrichment. This confirmed that lentiviral delivery is compatible with the system and enables stable, long-term expression of the reporter.

## Example 5

### Assay Based on Autologous B-Lymphoblastoid Cell Line (B-LCL) as Reporter Cells

- Autologous B-LCLs derived from human donors were transduced with lentiviral vectors encoding the FRET reporter and a library of human tumor antigens. T cells isolated from the same donor were co-cultured with the transduced B-LCLs, and FRET-shifted cells were sorted and sequenced. Epitopes derived from tumor-associated antigens such as MART-1 and NY-ESO-1 were identified, demonstrating that the method is functional in primary human cells and applicable to clinical settings.

## Definitions

- Antigen presenting cell refers to a cell capable of processing intracellular proteins and presenting peptide fragments on major histocompatibility complex class I molecules to cytotoxic T lymphocytes.

- Apoptosis refers to a programmed form of cell death characterized by cell shrinkage, membrane blebbing, chromatin condensation, and fragmentation into apoptotic bodies, often initiated by granzyme B-mediated proteolysis.

- Cytotoxic T-cell refers to a CD8+ T lymphocyte capable of recognizing peptide–MHC class I complexes and inducing target cell death through the release of perforin and granzymes.

- Epitope refers to a short peptide sequence derived from a protein that is presented by MHC class I molecules and recognized by a T-cell receptor.

- Effector agent refers to a molecule, such as granzyme B, released by cytotoxic T cells to induce apoptosis in target cells.

- Effector response refers to the biological outcome of T-cell recognition, including cytokine secretion, target cell lysis, and induction of apoptosis.

- Granzyme refers to a serine protease secreted by cytotoxic T cells and natural killer cells that cleaves intracellular substrates to induce apoptosis.

- Granzyme-perforin pathway refers to the mechanism by which cytotoxic T cells deliver granzymes into target cells via perforin-mediated membrane pores to initiate caspase-dependent and -independent apoptosis.

- Kit refers to a packaged set of reagents, cells, vectors, and instructions for performing the method of epitope discovery.

- Major histocompatibility complex refers to a group of cell surface proteins encoded by the HLA genes in humans that present peptides to T cells.

- Perforin refers to a pore-forming protein secreted by cytotoxic T cells that facilitates the entry of granzymes into target cells.

- Peptide refers to a short chain of amino acids, typically 8 to 12 residues in length, that can be presented by MHC class I molecules.

- Polymerase chain reaction refers to a method for amplifying specific DNA sequences using primers and a thermostable DNA polymerase.

- Primer refers to a short nucleic acid sequence used to initiate DNA synthesis during PCR or sequencing.

- Transgene refers to a gene or DNA sequence introduced into a cell or organism by genetic engineering.

- Transfection refers to the introduction of nucleic acids into eukaryotic cells using non-viral methods such as lipofection or electroporation.

- Transformation refers to the genetic alteration of a cell by uptake of exogenous DNA, commonly used in bacterial systems.

- Transduction refers to the introduction of nucleic acids into cells via viral vectors.

- Vector refers to a DNA molecule used as a vehicle to carry foreign genetic material into a host cell.

- Plasmid refers to a circular, double-stranded DNA molecule capable of autonomous replication in bacterial or eukaryotic cells.

- Phage vector refers to a bacteriophage-derived system used to deliver DNA into bacterial cells.

- Viral vector refers to a modified virus used to deliver genetic material into eukaryotic cells, including lentiviral, adenoviral, and retroviral systems.

- Bacterial vector refers to a plasmid or phage designed for propagation and manipulation in bacterial hosts.

- Episomal mammalian vector refers to a DNA construct that replicates independently of the host genome in mammalian cells.

- Non-episomal mammalian vector refers to a DNA construct that integrates into the host genome, such as lentiviral vectors.

- Recombinant expression vector refers to a DNA construct engineered to express a gene of interest in a host cell, typically containing a promoter, coding sequence, and regulatory elements.