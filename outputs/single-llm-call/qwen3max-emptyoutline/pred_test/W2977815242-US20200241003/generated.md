# DESCRIPTION

## BACKGROUND

The identification of T cell epitopes—short peptide fragments derived from antigens that are presented by major histocompatibility complex (MHC) molecules and recognized by T cell receptors (TCRs)—is a cornerstone of adaptive immunology and a critical step in the development of vaccines, diagnostics, and T cell-based immunotherapies. Despite decades of research, the systematic discovery of T cell epitopes remains a formidable scientific and technical challenge due to the extraordinary complexity of the biological systems involved. This complexity arises from multiple interrelated factors that collectively define the landscape of potential T cell recognition events.

First, the theoretical space of possible short peptides is astronomically large. Given that typical MHC class I–restricted epitopes range from 8 to 11 amino acids in length, and considering the 20 naturally occurring amino acids, the number of unique nonapeptides alone exceeds 5 × 10¹². While natural proteomes constrain this space significantly, the human proteome still contains millions of potential epitope candidates, and pathogen or tumor-derived proteomes can introduce additional layers of diversity through mutation, alternative splicing, or post-translational modifications. Exhaustively screening such vast sequence spaces using conventional methods is impractical.

Second, the genetic architecture of the MHC locus in humans—known as the human leukocyte antigen (HLA) system—is both polygenic and highly polymorphic. Individuals express multiple classical MHC class I (HLA-A, -B, -C) and class II (HLA-DR, -DQ, -DP) molecules, each encoded by separate genes. Moreover, each of these genes exists in hundreds to thousands of allelic variants across the global population. Critically, different HLA alleles exhibit distinct peptide-binding motifs, meaning that the same protein antigen may yield entirely different sets of presented epitopes in individuals with different HLA haplotypes. This genetic diversity necessitates personalized or population-tailored approaches to epitope discovery and complicates the generalization of findings across patient cohorts.

Third, the process by which intracellular proteins are converted into MHC-presented peptides—antigen processing—is neither random nor uniform. It involves a cascade of proteolytic events primarily mediated by the proteasome, followed by transport into the endoplasmic reticulum via the transporter associated with antigen processing (TAP), and final trimming by endoplasmic reticulum aminopeptidases (ERAPs). Each step introduces biases that influence which peptides are ultimately loaded onto MHC molecules. For example, certain protein domains may be resistant to proteasomal cleavage, or specific flanking sequences may enhance or inhibit peptide transport and trimming. Consequently, even peptides with high predicted MHC-binding affinity may never be naturally presented if they are not efficiently generated during antigen processing.

Fourth, the interaction between the TCR and the peptide–MHC (pMHC) complex is inherently dynamic and context-dependent. Unlike antibody–antigen interactions, which often exhibit high affinity and long half-lives, TCR–pMHC interactions are typically of low affinity (micromolar dissociation constants) and short duration. Yet, despite this apparent weakness, these interactions can trigger robust intracellular signaling cascades leading to T cell activation, cytokine production, and target cell killing. The outcome of TCR engagement depends not only on binding affinity but also on kinetic parameters such as on-rates and off-rates, the half-life of the pMHC complex itself, the density of pMHC on the target cell surface, and the involvement of co-receptors (e.g., CD8 or CD4) and adhesion molecules that stabilize the immunological synapse. Furthermore, TCRs are inherently cross-reactive, capable of recognizing multiple distinct pMHC complexes—a feature essential for immune coverage but problematic for precise epitope mapping.

In light of these challenges, two broad classes of epitope discovery methods have emerged: function-based assays and affinity-based assays. Function-based methods assess T cell responses in a physiological context by presenting candidate peptides on live antigen-presenting cells (APCs) and measuring downstream functional outputs such as cytokine secretion (e.g., interferon-γ ELISpot), upregulation of activation markers (e.g., CD137 or CD69), or direct target cell lysis. These assays preserve the natural antigen processing and presentation pathways and capture the integrated biological response of the T cell. However, they are inherently low-throughput because each candidate peptide must typically be tested individually or in small pools, followed by laborious deconvolution to identify the active component. Scaling these assays to proteome-wide levels is thus prohibitively resource-intensive.

Affinity-based methods, in contrast, bypass cellular physiology by directly measuring the physical interaction between soluble TCRs or T cells and synthetic pMHC complexes. Examples include pMHC multimer staining (e.g., tetramers or dextramers), single-chain trimer display, and combinatorial or barcoded pMHC libraries coupled with next-generation sequencing. These approaches offer high throughput and can screen millions of peptide variants in parallel. However, they suffer from a critical limitation: they do not account for whether a given peptide is actually processed and presented by cells in vivo. A high-affinity binder identified in vitro may be irrelevant in a biological context if it is never generated during natural antigen processing. Conversely, low-affinity epitopes that are efficiently presented and recognized in vivo may be missed.

Recent advances in TCR repertoire sequencing (TCR-seq) have further highlighted the need for scalable, function-based epitope discovery. Modern sequencing platforms routinely identify millions of unique TCR clonotypes from a single individual, revealing immense diversity and convergence across individuals. Public repositories now contain billions of TCR sequences. Yet, without knowledge of the antigens recognized by these TCRs, their biological significance remains obscure. The ability to screen defined T cell populations—whether monoclonal, oligoclonal, or polyclonal—against vast, unbiased libraries of potential epitopes in a functionally relevant manner is therefore a pressing unmet need.

Accordingly, there exists a significant gap in the art for a method that combines the physiological relevance of function-based assays with the scalability of affinity-based approaches. Such a method would enable the high-throughput identification of naturally processed and presented T cell epitopes that elicit genuine functional responses, thereby accelerating research in infectious disease, cancer immunotherapy, autoimmunity, and transplant rejection.

## SUMMARY OF THE INVENTION

The present invention provides a novel, high-throughput platform for the functional identification of T cell epitopes that overcomes the limitations of existing methodologies by integrating live-cell co-culture, granzyme B–mediated signal generation, fluorescence resonance energy transfer (FRET)-based detection, fluorescence-activated cell sorting (FACS), and deep amplicon sequencing. At the core of the invention is a genetically engineered reporter system expressed within antigen-presenting cells (APCs), which enables the selective isolation of those APCs that have been specifically targeted by cytotoxic T lymphocytes (CTLs) based on a measurable shift in fluorescent signal.

The invention is predicated on the biological principle that upon recognition of a cognate peptide–MHC complex on a target cell, activated CTLs deliver the serine protease granzyme B (GZMB) into the target cell cytoplasm via the perforin-dependent pathway. This delivery is highly specific and occurs only at the immunological synapse formed between the CTL and its target, minimizing off-target effects on bystander cells. The inventors have engineered a FRET-based reporter protein consisting of cyan fluorescent protein (CFP) and yellow fluorescent protein (YFP) linked by a peptide sequence that serves as a specific substrate for GZMB cleavage. In the uncleaved state, excitation of CFP results in efficient energy transfer to YFP, producing a characteristic FRET emission signature. Upon cleavage by GZMB, the physical separation of CFP and YFP abolishes FRET, leading to a concomitant increase in CFP emission and a decrease in YFP-associated FRET signal. This “FRET-shift” is readily detectable by flow cytometry and provides a robust, quantitative readout of CTL-mediated targeting.

To enable epitope discovery, the invention employs libraries of nucleic acid constructs—referred to herein as minigene libraries—wherein each construct encodes a short peptide candidate (typically 15–50 amino acids) flanking a putative epitope core. These minigene constructs are cloned into a lentiviral vector that also carries the GZMB-cleavable FRET reporter gene under the control of a constitutive promoter. Lentiviral transduction of MHC-matched APCs at a low multiplicity of infection (MOI) ensures that the majority of transduced cells express a single minigene along with the reporter. The resulting heterogeneous population of APCs, each displaying a unique peptide candidate in the context of its native MHC molecules, is then co-cultured with a population of T cells of interest—such as expanded tumor-infiltrating lymphocytes, vaccine-induced T cells, or TCR-transgenic T cells.

Following co-culture, APCs that present an epitope recognized by the T cells are selectively targeted, receive GZMB, and undergo cleavage of the FRET reporter. These “shifted” cells are isolated from the bulk population using FACS based on their altered fluorescence profile. Genomic DNA is then extracted from the sorted cells, and the integrated minigene sequences are amplified by PCR using primers specific to the conserved vector regions flanking the minigene insertion site. The resulting amplicons are subjected to high-throughput sequencing, allowing the identification of enriched minigene sequences that correspond to immunogenic epitopes.

The invention offers several key advantages over prior art. First, it operates within the natural antigen processing and presentation pathway, ensuring that identified epitopes are physiologically relevant. Second, it leverages the exquisite specificity of the granzyme–perforin system, minimizing false positives from bystander activation. Third, it is highly sensitive, capable of detecting epitope-specific APCs present at frequencies as low as 1 in 10,000 within a complex library. Fourth, it is compatible with polyclonal T cell populations, enabling the screening of biologically relevant samples such as tumor-infiltrating lymphocytes without the need for prior clonal isolation. Fifth, it does not require iterative panning or deconvolution, as epitopes can be identified directly from primary screens via sequencing.

In various embodiments, the invention may be applied to mouse or human systems, using a range of APC types including immortalized cell lines (e.g., EL4, ID8, B-LCLs) or engineered artificial APCs. The minigene libraries may be designed to cover entire pathogen or tumor proteomes, focus on regions of interest (e.g., mutated neoantigens), or consist of randomized sequences for de novo epitope discovery. The platform is particularly well-suited for applications in cancer immunotherapy (e.g., neoantigen validation), infectious disease (e.g., viral epitope mapping), autoimmunity (e.g., self-antigen identification), and basic T cell biology (e.g., TCR cross-reactivity profiling).

## DETAILED DESCRIPTION OF THE INVENTION

### Iterative Determination of T Cell Epitopes

The present invention provides a transformative approach to the iterative determination of T cell epitopes by enabling the simultaneous functional screening of thousands to millions of peptide candidates in a single experiment. Traditional epitope mapping strategies rely on sequential testing of overlapping peptides spanning a protein of interest, a process that is both time-consuming and resource-intensive. Even with pooling strategies, deconvolution of positive pools to identify the minimal epitope requires multiple rounds of testing. In contrast, the method of the present invention allows for the direct identification of immunogenic sequences from complex mixtures without the need for iterative refinement in many cases.

The iterative aspect of the invention arises not from repeated biological screening rounds but from the analytical depth afforded by next-generation sequencing. After a primary screen, sequencing data reveal not only the expected positive controls (if spiked in) but also a ranked list of enriched minigenes based on their differential abundance in the FRET-shifted versus unshifted populations. These candidates can then be subjected to bioinformatic filtering—for example, using algorithms such as NetMHCpan to predict MHC binding affinity or BLAST to assess homology to known proteins—and prioritized for secondary validation. In some embodiments, top candidates from a primary screen may be synthesized as a focused “panning library” and re-screened to confirm activity and eliminate false positives arising from sequencing or PCR artifacts. This secondary screen, however, is optional and often unnecessary due to the high specificity of the primary assay.

Moreover, the invention facilitates the mapping of minimal epitopes within longer minigene sequences. Because minigenes typically encode 15–50 amino acids, the exact core epitope recognized by the TCR may not be immediately apparent from the enriched sequence alone. However, by aligning multiple enriched minigenes that share a common core region, or by designing truncated variants of a hit minigene and retesting them in the assay, the minimal epitope can be rapidly delineated. This process is significantly accelerated compared to traditional alanine scanning or truncation series, as each variant can be tested in parallel within a new minigene library.

The iterative determination is further enhanced by the ability to incorporate feedback from biological context. For instance, when screening T cells isolated from a melanoma patient, enriched minigenes can be cross-referenced with the patient’s tumor exome to identify somatic mutations that generate neoantigens. Similarly, in infectious disease settings, hits can be mapped back to the pathogen genome to define protective epitopes. This closed-loop workflow—from library screening to sequence identification to biological validation—represents a significant advance in the speed and accuracy of epitope discovery.

### Reporter Cells

Reporter cells are central to the operation of the present invention and are engineered to serve dual functions: (1) to present candidate epitopes in the context of appropriate MHC molecules, and (2) to report T cell–mediated targeting through a quantifiable change in fluorescence. These cells are derived from antigen-presenting cell (APC) lines that are compatible with the T cell population being screened, ensuring MHC matching and functional antigen processing machinery.

In preferred embodiments, reporter cells are generated by stable transduction of APCs with a lentiviral vector encoding both a minigene (or minigene library) and a GZMB-cleavable FRET reporter. The lentiviral system ensures genomic integration and stable, long-term expression of the transgenes. Prior to use in co-culture assays, transduced cells undergo a purity sort by FACS to isolate a homogeneous population of cells that express the reporter at consistent levels and lack non-productive integrations (e.g., those containing stop codons or frameshifts in the minigene). This step is critical for reducing background noise and ensuring assay sensitivity.

The choice of APC line depends on the experimental context. For murine studies, commonly used lines include EL4 (a T cell lymphoma of C57BL/6 origin, H-2b haplotype) and ID8 (a syngeneic ovarian carcinoma line, also H-2b). Both lines express MHC class I molecules and are susceptible to CTL-mediated killing. EL4 cells, in particular, exhibit high MHC class I expression and robust FRET-shift signals, making them ideal for high-sensitivity screens. For human applications, autologous B-lymphoblastoid cell lines (B-LCLs) generated by Epstein-Barr virus (EBV) transformation of peripheral blood mononuclear cells (PBMCs) are preferred, as they provide a matched HLA background. Alternatively, engineered cell lines such as K562, which are normally MHC class I–negative, can be transduced to express specific HLA alleles of interest, creating customizable artificial APCs.

Reporter cells must maintain viability and normal cellular physiology throughout the assay. The FRET reporter itself is designed to be non-toxic and inert in the absence of GZMB, ensuring that baseline fluorescence does not interfere with cell health or antigen presentation. Following T cell co-culture, reporter cells that have received GZMB initiate apoptosis but remain intact for several hours, providing a “safe-sorting window” during which they can be recovered by FACS before membrane integrity is lost (as measured by propidium iodide exclusion). This temporal window is carefully calibrated—typically 4 to 8 hours post-co-culture—to maximize signal while preserving cell viability for downstream DNA extraction.

### Epitope-Encoding Nucleic Acid Libraries

Epitope-encoding nucleic acid libraries, or minigene libraries, are the molecular substrates that enable high-throughput epitope discovery in the present invention. Each minigene in the library encodes a short open reading frame (ORF) corresponding to a candidate epitope flanked by natural or optimized sequences that facilitate proteasomal processing and MHC loading. The design of these minigenes is critical to the success of the assay.

In one embodiment, minigenes are derived from known protein sequences, such as viral proteomes, tumor-associated antigens, or autoantigens. Overlapping minigenes tiling across a protein of interest can be synthesized as individual constructs or as a pooled library. In another embodiment, minigenes are designed to encode mutated neoantigens identified from tumor exome sequencing, with each minigene containing a single somatic mutation in its central region. In a third embodiment, randomized minigene libraries are constructed using degenerate oligonucleotides, enabling unbiased discovery of novel epitopes without prior knowledge of antigen source.

The length of the minigene ORF is optimized to balance efficient expression, processing, and presentation. Typically, minigenes encode 15 to 50 amino acids, which is sufficient to include the core epitope (8–11 aa for MHC class I) plus flanking residues that influence proteasomal cleavage. Shorter minigenes may be poorly processed, while longer ones risk introducing cryptic epitopes or reducing library complexity. The nucleic acid sequence is codon-optimized for the host species (e.g., mouse or human) to ensure high-level expression, and care is taken to avoid internal restriction sites that could interfere with cloning.

Libraries are cloned into a lentiviral transfer plasmid upstream of a P2A self-cleaving peptide sequence, which ensures stoichiometric co-expression of the minigene-encoded peptide and the downstream FRET reporter. The vector backbone includes a strong constitutive promoter (e.g., MNDU3) to drive high-level transcription. Library complexity is controlled during viral production by adjusting the amount of plasmid DNA and the scale of bacterial amplification. For primary screens, libraries of 10⁵ to 10⁶ unique minigenes are typical, though larger libraries are feasible with scaled-up cell culture.

Quality control of the library is performed by deep sequencing of the plasmid pool prior to viral production to assess diversity, representation, and the absence of dominant clones or contaminants. After transduction and purity sorting of reporter cells, a second sequencing check confirms that the cellular library accurately reflects the input plasmid library.

### Cytotoxic T-Cells

Cytotoxic T lymphocytes (CTLs) are the effector cells that drive the selection process in the present invention. These cells recognize peptide–MHC class I complexes on the surface of reporter cells and, upon engagement of their TCR with a cognate epitope, release granzyme B into the target cell, triggering the FRET-shift signal. The invention is compatible with a wide range of CTL sources, from monoclonal populations to complex polyclonal mixtures.

In validation studies, the inventors employed T cells from TCR-transgenic mouse strains such as OT-I (specific for SIINFEKL/H-2Kᵇ) and pmel-1 (specific for KVPRNQDWL/H-2Dᵇ). These monoclonal populations provide a high signal-to-noise ratio and are ideal for assay optimization. However, the true power of the invention lies in its ability to work with polyclonal T cell populations derived from biological samples. For example, tumor-infiltrating lymphocytes (TILs) expanded from melanoma or ovarian cancer patients can be screened directly against tumor antigen libraries to identify reactive neoantigens. Similarly, T cells from vaccinated individuals or convalescent patients can be used to map protective epitopes from pathogens.

CTLs are typically expanded ex vivo by stimulation with anti-CD3 and anti-CD28 antibodies in the presence of interleukin-2 (IL-2). This protocol generates a population of activated, proliferating CTLs that retain their antigen specificity and cytotoxic function. The expansion period is optimized to achieve sufficient cell numbers for screening while minimizing differentiation into exhausted or anergic states. In some cases, antigen-specific enrichment (e.g., by pMHC multimer sorting or cytokine capture) may be performed prior to expansion to increase the frequency of reactive clones.

The effector-to-target (E:T) ratio in co-culture is a critical parameter that influences assay sensitivity. Ratios ranging from 0.5:1 to 2:1 are typically used, with 1:1 being standard. Lower ratios may be necessary when working with limited T cell numbers (e.g., from TIL cultures), while higher ratios can enhance signal in low-frequency scenarios. Co-culture duration is also optimized—usually 4 to 8 hours—to allow sufficient time for GZMB delivery and FRET cleavage while avoiding excessive target cell death.

### Nucleic Acid Sequencing Techniques

Nucleic acid sequencing is the analytical engine that deciphers the output of the FRET-shift assay, transforming sorted cell populations into actionable epitope data. Following FACS isolation of FRET-shifted and unshifted reporter cells, genomic DNA is extracted and used as a template for PCR amplification of the integrated minigene cassettes. The primers used in this reaction are designed to anneal to conserved regions of the lentiviral vector that flank the minigene insertion site, ensuring that all library members are amplified with equal efficiency.

In preferred embodiments, sequencing libraries are prepared using a two-step PCR protocol. The first PCR uses vector-specific primers to amplify the minigene region, while the second PCR adds Illumina-compatible adapter sequences and sample-specific index barcodes, enabling multiplexing of multiple screens in a single sequencing run. This approach minimizes amplification bias and preserves the relative abundance of minigenes in the original cell populations.

Sequencing is performed on high-throughput platforms such as the Illumina MiSeq or NovaSeq, using paired-end chemistry (e.g., 2 × 250 bp) to ensure full coverage of the minigene insert. Raw reads are processed through a bioinformatic pipeline that includes quality filtering, read merging, error correction, and clustering to collapse PCR and sequencing duplicates. The resulting consensus sequences are then aligned to the reference library or analyzed de novo to identify enriched variants.

The key metric for epitope identification is the differential abundance (Δ abundance) of each minigene in the shifted versus unshifted populations. Enrichment is calculated as the log₂ fold-change or simple difference in relative frequency, and statistical thresholds (e.g., >10 standard deviations above the mean Δ abundance of all minigenes) are applied to distinguish true positives from background noise. This quantitative approach allows for the detection of rare epitopes even in highly complex libraries.

### Assessing Cellular Immunity to Specific Antigens

The invention provides a powerful tool for assessing cellular immunity to specific antigens in a variety of contexts. By screening T cell populations against defined minigene libraries, researchers can determine which epitopes within an antigen are immunodominant, subdominant, or cryptic. This information is invaluable for vaccine design, where the goal is often to elicit responses against protective epitopes while avoiding those that may drive immune escape or pathology.

For example, in cancer immunotherapy, the invention can be used to validate predicted neoantigens derived from tumor sequencing. Patient-derived TILs are screened against a library of minigenes encoding all somatic mutations identified in the tumor. Enriched minigenes correspond to true neoantigens that are naturally processed, presented, and recognized by the patient’s own T cells. These validated neoantigens can then be prioritized for inclusion in personalized cancer vaccines or for engineering into TCR-transgenic T cell therapies.

In infectious disease, the platform can map T cell responses to entire pathogen proteomes. Libraries covering all open reading frames of a virus (e.g., SARS-CoV-2, HIV, or influenza) can be screened with T cells from convalescent patients to identify conserved, protective epitopes that could serve as universal vaccine targets. Similarly, in autoimmune disorders, the invention can help identify self-antigens that drive pathogenic T cell responses, offering new targets for antigen-specific tolerance induction.

The assay also enables the study of T cell cross-reactivity—the ability of a single TCR to recognize multiple distinct epitopes. By screening a monoclonal T cell population against a diverse library, researchers can define the full “recognition repertoire” of that TCR, revealing potential off-target reactivities that could lead to autoimmunity or adverse events in adoptive T cell therapy.

### Further Applications

Beyond epitope discovery, the invention has numerous further applications in immunology and biotechnology. One such application is TCR validation and characterization. Soluble TCRs or TCR-transgenic T cells developed for therapeutic use can be screened against comprehensive self-antigen libraries to assess off-target reactivity and improve safety profiles.

Another application is immune monitoring. Longitudinal samples from patients undergoing immunotherapy (e.g., checkpoint blockade or adoptive T cell transfer) can be screened against tumor antigen libraries to track the evolution of T cell responses over time, correlating epitope reactivity with clinical outcomes.

The platform can also be adapted for MHC class II–restricted epitope discovery by using CD4⁺ T cells and APCs that express MHC class II molecules (e.g., dendritic cells or B-LCLs). Although the current reporter relies on GZMB, which is predominantly expressed by CD8⁺ T cells, alternative reporters based on other CTL effector molecules (e.g., granzyme A or perforin-induced calcium flux) could be developed for CD4⁺ T cell applications.

Additionally, the invention can be used to study the impact of antigen processing mutations. For instance, tumors with defects in the antigen processing machinery (e.g., TAP or β2-microglobulin loss) can be modeled by comparing epitope presentation in wild-type versus knockout APC lines.

### Kits

The invention may be embodied in commercial kits designed to facilitate adoption by researchers and clinicians. A typical kit would include: (1) a lentiviral transfer plasmid backbone containing the GZMB-cleavable FRET reporter and a multiple cloning site for minigene insertion; (2) packaging plasmids for lentivirus production (e.g., psPAX2 and pMD2.G); (3) protocols for viral production, APC transduction, and purity sorting; (4) primers for minigene amplification and sequencing library preparation; and (5) bioinformatic software for data analysis.

Optional components might include pre-made minigene libraries for common pathogens (e.g., influenza, CMV, EBV) or cancer types (e.g., melanoma, lung cancer), as well as matched APC lines and T cell expansion reagents. Kits could be tailored for mouse or human use, with human kits including HLA-typed B-LCLs or K562-based artificial APCs expressing common HLA alleles.

Such kits would democratize access to high-throughput epitope discovery, enabling laboratories without specialized expertise in virology or flow cytometry to perform sophisticated T cell antigen screening.

## Example 1

### Model Assays Based on Stably Transfected Mouse Lymphoblastic Cell Lines EL4 and EG7 as Model Reporter Cells

The utility of the FRET-shift assay was first demonstrated using the murine T lymphoblastic cell line EL4 and its derivative EG7, which stably expresses full-length ovalbumin. EL4 cells, derived from a C57BL/6 mouse thymoma, express high levels of H-2Kᵇ and H-2Dᵇ MHC class I molecules, making them ideal targets for CD8⁺ T cells restricted to these alleles. EG7 cells, generated by transfection of EL4 with an ovalbumin-expressing plasmid, naturally process and present the immunodominant SIINFEKL epitope (OVA₂₅₇–₂₆₄) in the context of H-2Kᵇ.

In initial experiments, EL4 cells were transduced with lentiviral vectors encoding either the OVA minigene (spanning OVA₂₄₁–₂₈₀, which includes SIINFEKL) or a scrambled control minigene (with the SIINFEKL sequence replaced by LKNFISEI). Both constructs also carried the GZMB-cleavable CFP-YFP FRET reporter. After purity sorting to isolate YFP⁺ cells with a resting FRET signature, the transduced EL4 populations were co-cultured with OT-I CD8⁺ T cells, which express a transgenic TCR specific for SIINFEKL/H-2Kᵇ.

Flow cytometric analysis revealed a robust FRET-shift in OVA minigene–expressing EL4 cells after 4 hours of co-culture with OT-I T cells, with over 60% of target cells shifting into the “Targeted” gate, compared to less than 2% in the scrambled control. The magnitude of the shift—defined as the change in FRET/CFP ratio—was significantly greater in EL4 cells than in ID8 ovarian carcinoma cells tested under identical conditions, likely due to higher MHC class I expression in EL4 (confirmed by flow cytometry). These results established EL4 as a highly sensitive reporter cell line for FRET-shift assays.

Subsequent experiments used EG7 cells as a positive control. When co-cultured with OT-I T cells, EG7 cells exhibited a strong FRET-shift, confirming that the assay detects naturally processed epitopes from full-length antigens as well as minigene-encoded fragments. This validated the physiological relevance of the platform and demonstrated its compatibility with both minigene and full-protein antigen sources.

## Example 2

### Confirming Function of Granzyme B-Sensitive Signal Generation Product

To confirm that the observed FRET-shift was specifically due to GZMB-mediated cleavage of the reporter, a series of control experiments were performed. First, the amino acid sequence of the linker between CFP and YFP was verified to match the optimal GZMB cleavage motif (IEPD). Mutagenesis of this motif to a non-cleavable sequence (e.g., IAPA) abolished the FRET-shift upon T cell co-culture, confirming GZMB dependence.

Second, co-cultures were treated with concanamycin A, a specific inhibitor of perforin pore formation. In the presence of concanamycin A, no FRET-shift was observed, demonstrating that GZMB delivery requires perforin-mediated entry into the target cell.

Third, OT-I T cells were pre-treated with Z-AAD-CMK, a cell-permeable GZMB inhibitor. These T cells failed to induce a FRET-shift in OVA-expressing targets, further confirming the role of GZMB.

Finally, time-course experiments showed that FRET-shift signal peaked at 6–8 hours post-co-culture, preceding the onset of propidium iodide positivity (a marker of late apoptosis) by 2–4 hours. This established a clear temporal window for safe cell sorting and confirmed that the assay detects early apoptotic events triggered specifically by GZMB.

## Example 3

### Model Assays Based On Stably Transfected Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

The ID8 murine ovarian carcinoma cell line was evaluated as an alternative reporter cell type. ID8 cells, derived from C57BL/6 mice, express moderate levels of H-2Kᵇ and are susceptible to CTL-mediated killing. ID8.G7-Ova cells, generated by stable transfection with an ovalbumin expression vector, naturally present SIINFEKL.

ID8 cells were transduced with OVA or scrambled minigene–FRET lentiviruses and co-cultured with OT-I T cells. A significant FRET-shift was observed in OVA-expressing cells (mean 35% shifted) versus scrambled controls (<1%), though the magnitude was lower than in EL4 cells, consistent with lower MHC expression. Mixed-population experiments, where OVA and scrambled cells were combined 1:1, showed >95% purity of OVA minigenes in the shifted gate by qPCR, confirming high specificity.

These results demonstrated that the assay functions effectively in epithelial-derived tumor lines, expanding its applicability to solid tumor models.

## Example 4

### Model Assays Based on Lentivirus-Transduced Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

Lentiviral transduction was compared to stable transfection for generating reporter cells. Lentivirally transduced ID8 cells showed more uniform reporter expression and higher transduction efficiency (>80% vs. ~50% for transfection). Purity sorting yielded a homogeneous population suitable for library screening. The kinetics and sensitivity of the FRET-shift were comparable between lentiviral and transfected lines, but lentiviral delivery was preferred for library applications due to scalability and consistency.

## Example 5

### Assay Based on Autologous B-Lymphoblastoid Cell Line (B-LCL) as Reporter Cells

To demonstrate human applicability, autologous B-LCLs were generated from a healthy donor by EBV transformation. The B-LCLs were transduced with a lentiviral vector encoding a CMV pp65-derived minigene (NLVPMVATV, HLA-A*02:01–restricted) and the FRET reporter. After purity sorting, the cells were co-cultured with HLA-matched CD8⁺ T cells expanded from the same donor and stimulated with CMV lysate.

A clear FRET-shift was observed in pp65 minigene–expressing B-LCLs compared to controls, confirming that the assay functions in human cells with endogenous antigen processing and presentation. This example validates the platform for clinical applications in personalized immunotherapy.

## Definitions

As used herein, the following terms have the meanings indicated:

“FRET-shift” refers to the change in fluorescence emission profile of a CFP-YFP fusion protein upon cleavage by granzyme B, characterized by a decrease in FRET signal and an increase in CFP signal.

“Minigene” means a nucleic acid sequence encoding a short peptide (15–50 amino acids) that includes a candidate T cell epitope and flanking residues.

“Reporter cell” is an antigen-presenting cell engineered to express a GZMB-cleavable FRET reporter and a minigene or minigene library.

“Cytotoxic T lymphocyte (CTL)” denotes a CD8⁺ T cell capable of recognizing peptide–MHC class I complexes and delivering granzyme B to target cells.

“Epitope” is a peptide fragment of an antigen that is presented by MHC molecules and recognized by a T cell receptor.

“Granzyme B (GZMB)” is a serine protease released by CTLs that cleaves after aspartic acid residues in target cells, initiating apoptosis.

“Lentiviral vector” is a gene delivery vehicle derived from lentiviruses, capable of stable genomic integration in dividing and non-dividing cells.

“Deep amplicon sequencing” refers to high-throughput sequencing of PCR-amplified DNA fragments to quantify sequence abundance in a population.