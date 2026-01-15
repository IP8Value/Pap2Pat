# DESCRIPTION

## BACKGROUND

T-cell epitopes are short peptide fragments derived from antigens that are presented on the surface of cells by major histocompatibility complex (MHC) molecules and recognized by T-cell receptors (TCRs). These epitopes play a central role in adaptive immune responses, enabling cytotoxic T lymphocytes (CTLs) to identify and eliminate infected or malignant cells. The identification and characterization of T-cell epitopes are therefore critical for understanding immune surveillance, developing vaccines, and designing immunotherapies for cancer, infectious diseases, and autoimmune disorders. Despite their importance, the discovery of T-cell epitopes remains a formidable challenge due to the immense complexity of the antigenic landscape and the biological constraints governing T-cell recognition.

MHC class I molecules, which are expressed on nearly all nucleated cells, present peptides derived from intracellular proteins to CD8+ cytotoxic T cells. These peptides are typically 8 to 11 amino acids in length and are generated through proteasomal degradation of endogenous proteins, followed by transport into the endoplasmic reticulum via the transporter associated with antigen processing (TAP). Within the endoplasmic reticulum, peptides bind to MHC class I molecules, and the resulting peptide–MHC (pMHC) complexes are trafficked to the cell surface for surveillance by T cells. The specificity of this interaction is governed by the polymorphic nature of MHC genes—known as human leukocyte antigen (HLA) genes in humans—which encode highly variable peptide-binding grooves. This genetic diversity results in distinct peptide-binding preferences across individuals, making epitope prediction and validation highly context-dependent.

The difficulties in T-cell epitope discovery are multifaceted. First, the theoretical space of possible peptide sequences is astronomically large, especially when considering the full proteome of pathogens or tumors. Second, the polygenic and polyallelic nature of HLA genes means that each individual expresses a unique set of MHC molecules, further complicating universal epitope identification. Third, not all peptides that bind MHC are immunogenic; factors such as intracellular protein abundance, proteolytic processing efficiency, pMHC stability, and co-receptor interactions influence whether a given pMHC complex will activate a T cell. Finally, TCR–pMHC interactions are inherently transient, low-affinity, and promiscuous, allowing a single TCR to recognize multiple epitopes—a phenomenon known as cross-reactivity—that defies simple biochemical modeling.

Conventional approaches to epitope discovery fall into two broad categories: function-based and affinity-based methods. Function-based assays measure T-cell activation in response to candidate antigens presented by antigen-presenting cells (APCs), using readouts such as cytokine secretion, proliferation, or target cell lysis. However, these methods are typically low-throughput, requiring individual testing of each candidate peptide in separate assays. Pooling strategies can increase throughput but necessitate laborious deconvolution to identify active components. Affinity-based methods, such as pMHC multimer staining or single-chain MHC display, bypass cellular processing and directly assess TCR binding to synthetic pMHC complexes. While scalable, these approaches fail to account for natural antigen processing, presentation dynamics, and the full biophysical context of T-cell activation, potentially yielding false positives or missing physiologically relevant epitopes.

Given these limitations, there is a pressing need for a high-throughput method that combines the physiological relevance of function-based assays with the scalability of affinity-based techniques. Such a method would enable unbiased screening of vast peptide libraries against polyclonal T-cell populations, facilitating the discovery of novel epitopes in complex biological contexts, including tumor microenvironments, chronic infections, and autoimmune lesions. The invention described herein addresses this unmet need by introducing a novel platform that leverages the granzyme–perforin pathway of cytotoxic T cells to selectively label and recover antigen-presenting cells that have been recognized by T cells, thereby enabling high-throughput identification of functional T-cell epitopes through next-generation sequencing.

## SUMMARY OF THE INVENTION

The present invention provides a high-throughput method for identifying T-cell epitopes that overcomes the limitations of conventional approaches by integrating functional T-cell recognition with scalable nucleic acid sequencing. The method utilizes engineered reporter cells that co-express candidate epitope-encoding nucleic acids and a signal-generating product that is specifically cleaved upon delivery of granzyme B from activated cytotoxic T cells. This cleavage event produces a detectable signal shift, enabling the selective isolation of reporter cells that have been targeted by T cells, followed by recovery and sequencing of the epitope-encoding nucleic acids they harbor.

In one embodiment, the reporter cells are genetically modified to express a Förster resonance energy transfer (FRET)-based fluorescent protein system comprising cyan fluorescent protein (CFP) and yellow fluorescent protein (YFP) linked by a granzyme B-sensitive peptide substrate. In the absence of T-cell recognition, the intact FRET construct emits a characteristic FRET signal upon excitation. Upon recognition by a cognate T cell, granzyme B is delivered into the reporter cell via the perforin pathway, cleaving the linker and disrupting FRET, thereby producing a measurable shift in fluorescence emission. This FRET-shift serves as a real-time, intrinsic indicator of T-cell-mediated targeting.

The method involves several key steps: first, a library of candidate epitope-encoding nucleic acids is introduced into a population of MHC-matched reporter cells, such that each cell expresses a single candidate epitope. The reporter cells are then co-cultured with a population of cytotoxic T cells of interest. Following co-culture, cells exhibiting a FRET-shift are isolated by fluorescence-activated cell sorting (FACS). The epitope-encoding nucleic acids from these sorted cells are recovered by PCR amplification and subjected to deep sequencing to identify the specific epitope sequences that elicited T-cell recognition.

An alternative method employs a leuco-dye-based signaling system, wherein granzyme B cleavage activates a colorimetric or fluorogenic signal, providing a complementary detection modality. In both configurations, the signal is generated intrinsically within the reporter cell, eliminating the need for external labeling or secondary assays.

A critical feature of the invention is the use of an enriched library of candidate epitope-encoding nucleic acids, which may be derived from pathogen genomes, tumor exomes, autoantigens, or randomized peptide sequences. These libraries are designed to encode peptides capable of being naturally processed and presented by MHC class I molecules, ensuring physiological relevance. The libraries may be constructed using degenerate codons to introduce sequence diversity, or synthesized based on in silico predictions of MHC binding.

The method is broadly applicable to assessing cellular immunity. By exposing reporter cells expressing a known antigen to a patient’s T cells, the presence and magnitude of antigen-specific immunity can be quantified. This enables diagnostic applications in infectious disease, cancer immunotherapy monitoring, and autoimmune disorder profiling.

Furthermore, the invention facilitates the identification of epitopes for vaccination. By screening T cells from convalescent individuals or vaccinated subjects against comprehensive antigen libraries, protective epitopes can be identified for inclusion in next-generation vaccines. Similarly, in autoimmune disorders, the method can uncover self-epitopes recognized by autoreactive T cells, guiding the development of tolerogenic therapies. For immune tolerance applications, the method can be used to identify epitopes that induce regulatory T-cell responses.

The invention also enables the identification of epitopes recognized by public T-cell clonotypes—TCRs that are shared across individuals and often associated with robust immune responses. By screening such clonotypes against large libraries, conserved epitopes can be discovered for universal vaccine design.

The expression of the epitope-encoding nucleic acid library is achieved through lentiviral transduction, ensuring stable integration and single-copy expression per cell at low multiplicity of infection. Detection and isolation of reporter cells are performed using high-speed FACS based on the FRET-shift or alternative signal. Finally, the isolated cells are analyzed by next-generation sequencing to decode the epitope sequences, completing the high-throughput discovery pipeline.

## DETAILED DESCRIPTION OF THE INVENTION

Conventional techniques for T-cell epitope discovery rely on either low-throughput functional assays or high-throughput affinity-based screens that lack physiological context. The present invention bridges this gap by introducing a function-based, high-throughput platform that preserves the natural antigen processing and presentation pathway while enabling scalable epitope identification. Central to this approach are engineered reporter cells that serve as both antigen-presenting platforms and intrinsic biosensors of T-cell recognition.

Reporter cells are mammalian cells, preferably of human or murine origin, that are genetically modified to express MHC class I molecules compatible with the T cells under investigation. MHC-matched reporter cells are essential to ensure that presented peptides are recognized in the correct immunological context. Autologous cells, such as B-lymphoblastoid cell lines (B-LCLs) derived from the same donor as the T cells, provide ideal histocompatibility. Alternatively, immortalized antigen-presenting cell lines, such as K562 cells engineered to express specific HLA alleles, offer a renewable and standardized platform.

These reporter cells are further modified to express a detectable signal-generating product that responds to T-cell-mediated cytotoxicity. The signal is designed to be activated specifically by granzyme B, a serine protease released by cytotoxic T cells upon recognition of cognate pMHC. Granzyme B enters the target cell via perforin pores and initiates apoptosis, but in the context of the invention, it also cleaves a reporter construct, generating a measurable signal before cell death occurs.

Reporter cell recognition by T cells triggers an effector response characterized by granzyme B release. This effector response is highly specific, as demonstrated by control experiments showing minimal bystander activation in mixed populations. Signal generation occurs rapidly, typically within 1 to 4 hours of co-culture, and peaks before the onset of irreversible apoptosis, creating a “safe-sorting window” during which targeted cells can be recovered intact for downstream analysis.

The method for determining T-cell epitopes involves the following steps: (1) construction of a library of nucleic acids encoding candidate epitopes; (2) delivery of the library into reporter cells at a low multiplicity of infection to ensure single epitope expression per cell; (3) co-culture of the reporter cell library with cytotoxic T cells of interest; (4) detection and isolation of reporter cells exhibiting a signal shift indicative of T-cell targeting; and (5) recovery and sequencing of the epitope-encoding nucleic acids from isolated cells.

High-throughput epitope screening is achieved by scaling the library size to hundreds of thousands or millions of unique sequences and using FACS to isolate rare targeted cells. Genetic modification of reporter cells is performed using viral vectors, particularly lentiviruses, which enable stable genomic integration and long-term expression. Schemes for high-throughput screening include iterative enrichment cycles, where primary hits are re-synthesized into a secondary library for re-screening to confirm specificity and reduce false positives.

Bioinformatics methods are employed to analyze sequencing data, including quality control, error correction, clustering of similar sequences, and alignment to reference databases. Iterative determination of T-cell epitopes enhances sensitivity and specificity by focusing subsequent rounds on enriched candidates.

The enriched library of candidate epitope-encoding nucleic acids may vary in size and composition. Exemplary libraries contain between 10⁴ and 10⁶ unique sequences, with representation frequencies calibrated to ensure sufficient clonal redundancy for detection. Libraries may be derived from cDNA or genomic DNA of diseased tissues, pathogen genomes, or designed de novo using computational predictions.

### Iterative Determination of T Cell Epitopes

The iterative determination of T-cell epitopes is a refinement strategy that improves the signal-to-noise ratio in epitope discovery. The method begins with a primary screen of a diverse library against T cells of interest. Reporter cells exhibiting a signal shift are isolated, and their epitope-encoding sequences are identified by sequencing. The top enriched sequences—those significantly above background—are then synthesized into a secondary, focused library. This secondary library is reintroduced into fresh reporter cells and subjected to a second round of screening under identical conditions. Sequences that are consistently enriched across both rounds are considered high-confidence epitopes.

The steps of the iterative method include: (1) performing an initial high-diversity screen; (2) identifying putative epitope sequences based on enrichment metrics (e.g., >10 standard deviations above background); (3) synthesizing a secondary library comprising the top N candidates (e.g., N = 480); (4) repeating the co-culture and sorting procedure; and (5) validating reproducibility of enrichment. This cycle may be repeated multiple times to achieve high specificity.

The identifying step relies on statistical thresholds derived from control experiments, ensuring that only biologically relevant signals are advanced. Repeating cycles effectively filters out stochastic noise and non-specific binders. Exemplary frequencies show that true epitopes maintain or increase enrichment in secondary screens, while false positives diminish.

### Reporter Cells

Reporter cells are engineered to possess dual functionality: antigen presentation and signal generation. Their capabilities include stable expression of MHC class I molecules, efficient processing and presentation of minigene-encoded peptides, and production of a cleavable reporter construct. The inclusion of MHC class I is motivated by the need to present peptides to CD8+ T cells in a physiologically relevant manner.

Autologous cells, such as B-LCLs, are preferred for human applications due to perfect HLA matching. Immortalized antigen-presenting cell lines, like EL4 (murine) or ID8 (murine ovarian cancer), are used in model systems. Transfection and transduction are employed to deliver genetic constructs, with lentiviral vectors offering superior efficiency and stability.

The signal-generating product is a fusion protein designed to be cleaved by granzyme B. In the FRET-based system, CFP and YFP are linked by a peptide substrate containing the consensus granzyme B cleavage motif (Ile-Glu-Pro-Asp). Cleavage separates the fluorophores, abolishing FRET and increasing CFP emission. Alternative leuco-dye-based systems use enzyme-activatable chromogens that become fluorescent upon cleavage.

Exemplary signaling systems include the CFP-YFP FRET pair, as well as other FRET pairs such as mTurquoise2-sYFP2. The genetic construct is cloned into a lentiviral transfer plasmid under a strong promoter (e.g., MNDU3), with a P2A self-cleaving peptide ensuring stoichiometric expression of the epitope and reporter. Lentivirus constructs are produced by co-transfection of HEK293T cells with packaging and envelope plasmids.

Human cell lines suitable for reporter cell engineering include K562, HEK293, and primary fibroblasts, all of which can be modified to express desired HLA alleles and the reporter system.

### Epitope-Encoding Nucleic Acid Libraries

Epitope-encoding nucleic acid libraries are collections of DNA or RNA sequences designed to encode peptides that can be processed and presented by MHC class I. These libraries vary in size, from small focused sets to large randomized pools exceeding 10⁶ members. The encoded peptides are typically 8–40 amino acids in length, encompassing minimal epitopes and flanking sequences to facilitate natural processing.

Libraries may be constructed by synthesizing overlapping peptide segments from a protein of interest, such as a tumor antigen or viral protein. Degenerate codons (e.g., NNK) are used to introduce diversity at specific positions, enabling exploration of sequence space. Coding segments are ligated into expression vectors, such as lentiviral backbones, using restriction sites or Gibson assembly.

Libraries can be prepared from cDNA or genomic DNA isolated from an individual, capturing patient-specific antigens. In cancer, libraries may be derived from tumor exome sequencing or neoantigen prediction pipelines. In silico T-cell epitope prediction tools, such as NetMHCpan, guide the selection of candidate sequences to enrich for potential binders. Whole-genome sequencing of an individual can also provide the source material for personalized epitope libraries.

### Cytotoxic T-Cells

Cytotoxic T cells (CTLs) are obtained from various sources, including peripheral blood, lymphoid organs, or diseased tissues. In cancer, tumor-infiltrating lymphocytes (TILs) are a rich source of antigen-specific T cells. TILs are obtained by excising tumor specimens, mechanically and enzymatically disaggregating the tissue, and isolating CD8+ T cells by FACS or magnetic beads.

TILs are expanded in vitro using anti-CD3/CD28 stimulation in the presence of interleukin-2 (IL-2). Microcultures are established from single-cell digests to maintain clonal diversity. TIL activity and specificity are assessed by measuring cytokine secretion, particularly interferon-gamma (IFN-γ), using ELISA or intracellular staining. Stimulation with autologous tumor cell lines confirms reactivity.

Multiple original wells are used to preserve independent T-cell clones, and cultures are maintained separately to avoid dominance by fast-growing clones. Specificity is further validated by tetramer staining or functional assays against known antigens.

### Nucleic Acid Sequencing Techniques

The nucleotide sequences of epitope-encoding nucleic acids are determined using commercial DNA sequencers, primarily Illumina platforms. Amplicon libraries are prepared by PCR using primers flanking the minigene insertion site, with indexed adapters for multiplexing. Paired-end sequencing (e.g., 2×250 bp) ensures accurate reconstruction of full-length inserts. Bioinformatic pipelines assemble reads, correct errors, and quantify sequence frequencies.

### Assessing Cellular Immunity to Specific Antigens

To test an individual’s cellular immunity to a specific antigen, reporter cells expressing that antigen are exposed to the individual’s T cells. The proportion of reporter cells undergoing signal shift correlates with the frequency and avidity of antigen-specific T cells, providing a quantitative measure of cellular immunity.

### Further Applications

The method identifies T-cell–antigen interactions in cancer, infection, and autoimmunity. It guides cancer vaccine design by revealing immunogenic neoantigens and supports autologous cell therapy by selecting reactive T-cell clones. In transplantation, it improves donor–recipient matching by assessing alloreactive T-cell responses.

### Kits

Kits include mammalian reporter cells pre-engineered with the signal-generating system, lentiviral vectors for delivering epitope libraries, and instructions for transduction and co-culture. Additional components may include FACS gating controls, PCR primers, and sequencing adapters.

## Example 1

### Model Assays Based on Stably Transfected Mouse Lymphoblastic Cell Lines EL4 and EG7 as Model Reporter Cells

Cytotoxic T lymphocytes (CTLs) derived from OT-I TCR transgenic mice were co-cultured with EL4 or EG7 (EL4 expressing ovalbumin) reporter cells stably transfected with the FRET reporter construct. EG7 cells, which present the SIINFEKL epitope, induced significant FRET-shift in reporter cells, whereas parental EL4 did not. Apoptosis was confirmed by propidium iodide staining, which occurred after peak FRET-shift, validating the safe-sorting window.

## Example 2

### Confirming Function of Granzyme B-Sensitive Signal Generation Product

Reporter cells expressing the CFP-linker-YFP construct were treated with recombinant granzyme B and perforin. Rapid loss of FRET signal and increase in CFP emission confirmed that the linker is specifically cleaved by granzyme B, validating the molecular design of the signal-generating product.

## Example 3

### Model Assays Based On Stably Transfected Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

ID8 cells expressing either Ova minigene or scrambled control were mixed 1:1 and co-cultured with OT-I CTLs. FACS sorting based on FRET-shift isolated targeted cells, which were >95% Ova-positive by qPCR. Deep sequencing confirmed enrichment of the SIINFEKL-encoding minigene, demonstrating high specificity.

## Example 4

### Model Assays Based on Lentivirus-Transduced Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

Lentiviral transduction of ID8 cells with Ova minigene-FRET constructs yielded stable reporter lines. Co-culture with OT-I CTLs produced robust FRET-shift, comparable to stably transfected lines, confirming that lentiviral delivery is a viable and scalable alternative.

## Example 5

### Assay Based on Autologous B-Lymphoblastoid Cell Line (B-LCL) as Reporter Cells

Autologous B-LCLs from a human donor were transduced with a viral epitope library and co-cultured with donor-derived CTLs. FRET-shifted cells were isolated and sequenced, identifying known viral epitopes, demonstrating applicability to human systems.

## Definitions

An antigen presenting cell (APC) is a cell that displays antigen complexed with MHC molecules to T cells. Apoptosis is programmed cell death mediated by caspases and granzymes. A cytotoxic T-cell is a CD8+ T lymphocyte that kills infected or malignant cells. An epitope is the specific part of an antigen recognized by a TCR. An effector agent is a molecule, such as granzyme B, that mediates T-cell function. An effector response is the functional outcome of T-cell activation, such as target cell killing. Granzyme is a serine protease released by cytotoxic lymphocytes. The granzyme-perforin pathway is the mechanism by which granzymes enter target cells. A kit is a packaged set of reagents for performing the invention. Major histocompatibility complex (MHC) is a set of cell surface proteins essential for antigen presentation. Perforin is a pore-forming protein that facilitates granzyme entry. A peptide is a short chain of amino acids. Polymerase chain reaction (PCR) is a method to amplify DNA. A primer is a short nucleic acid sequence that initiates DNA synthesis. A transgene is an exogenous gene introduced into a cell. Transfection is non-viral delivery of nucleic acids. Transformation is genetic alteration of a cell. Transduction is viral delivery of nucleic acids. A vector is a DNA molecule used to carry foreign genetic material. A plasmid is a circular double-stranded DNA vector. A phage vector is derived from bacteriophage. A viral vector is based on a virus. A bacterial vector replicates in bacteria. An episomal mammalian vector persists without integration. A non-episomal mammalian vector integrates into the genome. A recombinant expression vector contains regulatory elements for gene expression.