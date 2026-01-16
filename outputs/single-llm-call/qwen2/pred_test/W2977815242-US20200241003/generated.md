# DESCRIPTION

## BACKGROUND

Identifying T-cell epitopes is a challenging task due to the complexity of antigen-specific T-cell activation. Several factors contribute to this complexity. First, the vast number of unique short peptides that could exist creates an immense T-cell epitope space to be searched. Second, peptide-presenting major histocompatibility complex (MHC) molecules are encoded in humans by highly polymorphic HLA genes, leading to different MHC variants with distinct peptide-binding and T-cell receptor (TCR)-binding preferences. Third, variations in the intracellular expression level of antigenic proteins and biases in proteolytic processing also influence pMHC immunogenicity. Finally, TCR/pMHC interactions are transient, promiscuous, and relatively low-affinity compared to epitope recognition by antibodies.

Various function-based and affinity-based methods of antigen screening are currently in use. Function-based methods involve presenting candidate T-cell peptides on target cell surfaces and testing their ability to generate functional T-cell responses, identified through T cell-based read-outs such as cytokine release, activation of an NFAT-linked reporter, or monitoring the destruction of antigen-presenting cells (APCs) to measure functional T-cell activation. These methods typically require individual candidate antigens to be tested "one-by-one" in separate reactions, which limits their scalability. Pooling strategies can increase the search space but require subsequent deconvolution, making them laborious.

Affinity-based methods, such as single-chain MHC display or combinatorial/barcoded pMHC-multimer surface staining, seek to circumvent many of the limitations of function-based methods. While scalable, these methods bypass natural antigen processing, presentation, and T-cell activation, relying solely on TCR/pMHC affinity as a proxy for T-cell recognition. This can lead to the identification of high-affinity epitopes that are physiologically irrelevant and the missing of low-affinity but physiologically important epitopes.

New methods that combine the strengths of function-based approaches with the scalability of affinity-based approaches are essential for advancing our understanding of T-cell biology. TCR sequencing (TCR-seq) studies are routinely used to reveal millions of unique TCR α- and/or β-chains per individual and investigate T-cell repertoires. However, TCR-seq data do not provide information about the specific antigenic determinants of these clonotypes. Therefore, rational screening of T-cell populations against vast and unbiased libraries of peptides is crucial to reveal the landscape of epitopes they recognize.

## SUMMARY OF THE INVENTION

The present invention provides a method for high-throughput function-based T-cell antigen discovery. The method is based on a fundamentally different design from classic T-cell activity assays, wherein the main concept is the co-expression of candidate epitopes, encoded as minigenes in APCs, and a reporter system that is intrinsic to the APC instead of the T cell. This configuration allows for targeted APCs to become distinguishable from non-targeted APCs in T-cell co-cultures and facilitates the selective recovery of immunogenic antigen-bearing cells from irrelevant bystanders in the bulk population.

To realize this design, large libraries of short-peptide-coding DNA minigene sequences are cloned into a lentiviral transfer plasmid alongside a reporter system that is sensitive to the granzyme B (GZMB) protease. Infection of sero-matched target cells with these libraries is performed at a multiplicity of infection (MOI) that favors a single minigene per APC. Transduced target cells are co-incubated with expanded cytotoxic T lymphocyte (CTL) populations of interest. After co-culture, fluorescence-activated cell sorting (FACS) is conducted based on the FRET-shift status to isolate only target cells carrying putatively antigenic minigenes. Recovered minigenes are PCR-amplified and sequenced to reveal the epitopes eliciting reactivity from the screened CTL.

## DETAILED DESCRIPTION OF THE INVENTION

### Iterative Determination of T Cell Epitopes

The method for iterative determination of T cell epitopes involves several key steps. First, large libraries of short-peptide-coding DNA minigenes are generated and cloned into a lentiviral transfer plasmid. These minigenes are designed to encode candidate epitopes and are co-expressed with a reporter system that is sensitive to GZMB. The reporter system consists of a fusion protein of cyan fluorescent protein (CFP) and yellow fluorescent protein (YFP) moieties separated by a peptide linker that acts as a cleavage substrate for GZMB. When fused, the CFP-YFP reporter protein produces a FRET signal upon excitation with violet light. Cleavage of the fusion protein by GZMB causes a loss of FRET signal and concomitant rescue of free CFP signal, resulting in a FRET-shift that can be monitored in FACS.

Target cells are infected with the lentiviral minigene libraries at an MOI that ensures a single minigene per APC. These transduced target cells are then co-incubated with expanded CTL populations. After co-culture, FACS is performed to isolate target cells that have undergone FRET-shift, indicating that they have been recognized and targeted by the CTL. The isolated cells are lysed, and the minigenes are recovered by PCR and sequenced to identify the epitopes that elicited T-cell reactivity.

### Reporter Cells

Reporter cells are engineered to express a GZMB-cleavable reporter protein. The reporter protein is a fusion of CFP and YFP moieties separated by a peptide linker that serves as a cleavage substrate for GZMB. In the absence of GZMB, the reporter protein emits a resting FRET signature when excited with violet light. Upon entry of GZMB into the cell, the reporter is cleaved, causing a loss of FRET signal and a concomitant rescue of free CFP signal. This FRET-shift is easily distinguishable in FACS, allowing for the isolation of cells undergoing T-cell targeting.

### Epitope-Encoding Nucleic Acid Libraries

Epitope-encoding nucleic acid libraries are generated by synthesizing degenerate oligonucleotides containing a stretch of randomized bases. These oligonucleotides are amplified by PCR and ligated into the minigene site of a lentiviral transfer plasmid. The plasmid ligation products are transformed into electrocompetent bacteria and amplified on solid agar to obtain a diverse library of random minigenes. The plasmid DNA is isolated and used to generate lentivirus, which is then used to transduce target cells at an MOI favoring one insertion event per cell.

### Cytotoxic T-Cells

Cytotoxic T-cells (CTLs) are expanded by anti-CD3/28 stimulation from splenocytes of transgenic mice or wild-type mice. The expanded CTLs are co-cultured with the transduced target cells to test for T-cell recognition of the candidate epitopes. The co-culture is maintained for a period that allows for the detection of FRET-shift in the reporter cells, indicating T-cell targeting.

### Nucleic Acid Sequencing Techniques

Nucleic acid sequencing techniques are used to characterize the recovered minigenes from the FRET-shifted cells. Genomic DNA is isolated from the sorted cells and used as a template for PCR amplification of the integrated minigenes. The amplicons are sequenced using next-generation sequencing platforms, such as the Illumina MiSeq, to identify the epitopes that elicited T-cell reactivity.

### Assessing Cellular Immunity to Specific Antigens

The method allows for the assessment of cellular immunity to specific antigens by screening large libraries of candidate epitopes. The FRET-shift/amplicon-sequencing approach can detect relevant antigenic minigenes even when present at low frequencies, such as 1 in 10,000. This sensitivity is achieved by leveraging the natural antigen processing and presentation pathway in target cells and maintaining the biophysical context of natural TCR/pMHC receptor-ligand interactions.

### Further Applications

The method has several further applications, including the study of cellular immunity, supplementing the existing T-cell epitope knowledge base, and guiding the development of novel T cell-based immunotherapies. The approach can be applied to various biological contexts, such as cancer, autoimmunity, and infectious diseases, to identify epitopes recognized by T cells and understand the landscape of T-cell receptor/epitope interactions.

### Kits

Kits for performing the FRET-shift/amplicon-sequencing method are provided. The kits include components such as the lentiviral transfer plasmid, reporter cells, and reagents for viral transduction, cell sorting, and nucleic acid sequencing. The kits are designed to facilitate the high-throughput screening of T-cell epitopes and the identification of immunogenic antigens.

## Example 1

### Model Assays Based on Stably Transfected Mouse Lymphoblastic Cell Lines EL4 and EG7 as Model Reporter Cells

In this example, the murine ovarian cancer cell line, ID8, was used as APC for the well-characterized model TCR, OT-I. ID8 cells were transduced with minigene constructs coding for a 40 amino acid stretch of the chicken ovalbumin protein (OVAL241-280) with either the intact OT-I minimal epitope (SIINFEKL) or a scrambled version of this epitope (LKNFISEI). CD8+ T cells were expanded by anti-CD3/28 stimulation from splenocytes of the OT-I TCR-transgenic mouse and co-cultured with each target cell line separately. Flow-cytometric analyses of co-cultures indicated that SIINFEKL+ cells underwent significant and substantial cleavage of their encoded reporter protein relative to the scrambled negative control. These data provide evidence that the FRET-shift assay described herein is capable of efficiently detecting target cells harboring the correct antigen.

## Example 2

### Confirming Function of Granzyme B-Sensitive Signal Generation Product

To confirm the function of the GZMB-sensitive reporter gene, the murine ovarian cancer cell line, ID8, was used as APC for the well-characterized model TCR, OT-I. ID8 cells were transduced with minigene constructs coding for a 40 amino acid stretch of the chicken ovalbumin protein (OVAL241-280) with either the intact OT-I minimal epitope (SIINFEKL) or a scrambled version of this epitope (LKNFISEI). CD8+ T cells were expanded by anti-CD3/28 stimulation from splenocytes of the OT-I TCR-transgenic mouse and co-cultured with each target cell line separately. Flow-cytometric analyses of co-cultures indicated that SIINFEKL+ cells underwent significant and substantial cleavage of their encoded reporter protein relative to the scrambled negative control. These data provide evidence that the FRET-shift assay described herein is capable of efficiently detecting target cells harboring the correct antigen.

## Example 3

### Model Assays Based On Stably Transfected Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

In this example, the murine ovarian cancer cell line, ID8, was used as APC for the well-characterized model TCR, OT-I. ID8 cells were transduced with minigene constructs coding for a 40 amino acid stretch of the chicken ovalbumin protein (OVAL241-280) with either the intact OT-I minimal epitope (SIINFEKL) or a scrambled version of this epitope (LKNFISEI). CD8+ T cells were expanded by anti-CD3/28 stimulation from splenocytes of the OT-I TCR-transgenic mouse and co-cultured with each target cell line separately. Flow-cytometric analyses of co-cultures indicated that SIINFEKL+ cells underwent significant and substantial cleavage of their encoded reporter protein relative to the scrambled negative control. These data provide evidence that the FRET-shift assay described herein is capable of efficiently detecting target cells harboring the correct antigen.

## Example 4

### Model Assays Based on Lentivirus-Transduced Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells

In this example, the murine ovarian cancer cell line, ID8, was used as APC for the well-characterized model TCR, OT-I. ID8 cells were transduced with lentivirus containing minigene constructs coding for a 40 amino acid stretch of the chicken ovalbumin protein (OVAL241-280) with either the intact OT-I minimal epitope (SIINFEKL) or a scrambled version of this epitope (LKNFISEI). CD8+ T cells were expanded by anti-CD3/28 stimulation from splenocytes of the OT-I TCR-transgenic mouse and co-cultured with each target cell line separately. Flow-cytometric analyses of co-cultures indicated that SIINFEKL+ cells underwent significant and substantial cleavage of their encoded reporter protein relative to the scrambled negative control. These data provide evidence that the FRET-shift assay described herein is capable of efficiently detecting target cells harboring the correct antigen.

## Example 5

### Assay Based on Autologous B-Lymphoblastoid Cell Line (B-LCL) as Reporter Cells

In this example, autologous B-lymphoblastoid cell lines (B-LCL) were used as APC for the well-characterized model TCR, OT-I. B-LCL cells were transduced with minigene constructs coding for a 40 amino acid stretch of the chicken ovalbumin protein (OVAL241-280) with either the intact OT-I minimal epitope (SIINFEKL) or a scrambled version of this epitope (LKNFISEI). CD8+ T cells were expanded by anti-CD3/28 stimulation from splenocytes of the OT-I TCR-transgenic mouse and co-cultured with each target cell line separately. Flow-cytometric analyses of co-cultures indicated that SIINFEKL+ cells underwent significant and substantial cleavage of their encoded reporter protein relative to the scrambled negative control. These data provide evidence that the FRET-shift assay described herein is capable of efficiently detecting target cells harboring the correct antigen.

## Definitions

- **Antigen**: A substance that induces an immune response, particularly the production of antibodies or the activation of T cells.
- **Cytotoxic T Lymphocyte (CTL)**: A type of T cell that can directly kill cells infected by pathogens or cancer cells.
- **Epitope**: The part of an antigen that is recognized by the immune system, specifically by antibodies, B cells, or T cells.
- **Fluorescence Resonance Energy Transfer (FRET)**: A physical process where energy is transferred from one fluorophore (donor) to another (acceptor) without the emission of a photon.
- **Granzyme B (GZMB)**: A serine protease released by cytotoxic T cells and natural killer cells to induce apoptosis in target cells.
- **Lentivirus**: A genus of retroviruses that can cause chronic and deadly diseases in their hosts, often used as a vector in gene therapy and research.
- **Major Histocompatibility Complex (MHC)**: A group of genes that code for proteins responsible for the regulation of the immune system and the recognition of foreign substances.
- **Minigene**: A synthetic gene construct that encodes a short peptide or protein, often used in genetic studies and gene therapy.
- **Multiplexity of Infection (MOI)**: The ratio of infectious particles (such as viruses) to target cells in a culture.
- **Polymerase Chain Reaction (PCR)**: A technique used to amplify specific DNA sequences, enabling the detection and quantification of genetic material.
- **T-Cell Receptor (TCR)**: A protein complex on the surface of T cells that recognizes and binds to antigens presented by MHC molecules.
- **Tumor-Infiltrating Lymphocytes (TIL)**: Immune cells that have migrated into a tumor, often indicative of an immune response against the cancer.