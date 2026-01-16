Below is the drafted patent application following the provided outline and research paper. The application is written in formal patent language and maintains the structure specified in the outline.

---

# DESCRIPTION  

## BACKGROUND  

The identification of T-cell epitopes is a critical yet challenging task in immunology due to the immense complexity of antigen-specific T-cell activation. The vast number of potential peptide sequences, the polygenic and polyallelic nature of human leukocyte antigen (HLA) genes, variations in intracellular antigen processing, and the low-affinity, transient nature of T-cell receptor (TCR) and peptide-major histocompatibility complex (pMHC) interactions contribute to this challenge. Current methods for T-cell epitope identification are broadly categorized into function-based and affinity-based approaches, each with significant limitations.  

Function-based methods involve presenting candidate peptides on target cell surfaces and measuring T-cell responses through cytokine release, reporter activation, or antigen-presenting cell (APC) destruction. However, these methods require laborious one-by-one testing of individual peptides, making them impractical for large-scale epitope screening. Affinity-based techniques, such as single-chain MHC display or combinatorial pMHC-multimer staining, bypass natural antigen processing and rely solely on TCR/pMHC binding affinity. While scalable, these methods may yield physiologically irrelevant epitopes due to their inability to account for critical biophysical parameters influencing T-cell activation.  

There is a pressing need for novel methodologies that combine the physiological relevance of function-based assays with the scalability of affinity-based techniques. Such advancements would facilitate comprehensive T-cell epitope discovery, enabling deeper insights into T-cell biology and the development of targeted immunotherapies for cancer, infectious diseases, and autoimmune disorders.  

## SUMMARY OF THE INVENTION  

The present invention provides a high-throughput, function-based method for identifying T-cell epitopes by leveraging the specificity of the granzyme-perforin pathway. The method employs a reporter system intrinsic to antigen-presenting cells (APCs) rather than T cells, enabling the selective recovery of immunogenic antigen-bearing cells from non-targeted bystanders.  

Key components of the invention include:  
1. **Reporter Cells**: APCs engineered to express a Förster resonance energy transfer (FRET)-based reporter protein sensitive to granzyme B (GZMB) cleavage.  
2. **Epitope-Encoding Nucleic Acid Libraries**: Lentivirally delivered minigene libraries encoding diverse peptide sequences for presentation on APCs.  
3. **Cytotoxic T-Cells (CTLs)**: Expanded populations of CTLs of interest are co-cultured with reporter cells to identify immunogenic epitopes.  
4. **Nucleic Acid Sequencing Techniques**: High-throughput sequencing of recovered minigenes to identify epitopes eliciting T-cell reactivity.  

The invention enables scalable, physiologically relevant epitope discovery by combining natural antigen processing and presentation with high-throughput sequencing, overcoming limitations of existing methods.  

## DETAILED DESCRIPTION OF THE INVENTION  

### Iterative Determination of T Cell Epitopes  

The invention provides a method for iteratively determining T-cell epitopes by co-culturing reporter cells expressing epitope-encoding minigene libraries with CTLs of interest. Upon recognition of a cognate epitope, CTLs deliver GZMB to the reporter cell, cleaving the FRET reporter and generating a detectable signal shift. Reporter cells exhibiting this shift are isolated, and the integrated minigenes are recovered and sequenced to identify the immunogenic epitopes. This iterative process allows for the screening of vast peptide libraries while maintaining physiological relevance.  

### Reporter Cells  

Reporter cells are engineered to express a FRET-based reporter protein comprising cyan fluorescent protein (CFP) and yellow fluorescent protein (YFP) moieties linked by a GZMB-cleavable peptide substrate. In the absence of GZMB, the reporter emits a FRET signal upon excitation with violet light. Upon GZMB cleavage, the FRET signal is lost, and free CFP emission is rescued, enabling flow cytometric detection and isolation of targeted cells. Suitable reporter cells include murine cell lines (e.g., ID8, EL4) and human cell lines (e.g., K562-based artificial APCs), modified to express the reporter system.  

### Epitope-Encoding Nucleic Acid Libraries  

The invention utilizes nucleic acid libraries encoding diverse peptide sequences for presentation on reporter cells. These libraries are constructed by cloning degenerate oligonucleotides or defined peptide-coding sequences into lentiviral transfer plasmids alongside the FRET reporter. Lentiviral transduction ensures single minigene integration per cell, facilitating unambiguous epitope identification. Libraries may span entire proteomes or focus on specific antigen sets, enabling tailored epitope discovery.  

### Cytotoxic T-Cells  

CTLs used in the invention may be derived from transgenic models, polyclonal populations, or clinical samples. Prior to co-culture, CTLs are activated and expanded using anti-CD3/CD28 stimulation and interleukin-2 (IL-2). The method is compatible with mixed CTL populations, allowing for epitope discovery in polyclonal contexts, such as tumor-infiltrating lymphocytes (TILs) or peripheral blood mononuclear cells (PBMCs).  

### Nucleic Acid Sequencing Techniques  

Recovered minigenes from targeted reporter cells are amplified using PCR with Illumina adapter-tailed primers and sequenced on high-throughput platforms (e.g., Illumina MiSeq). Bioinformatics pipelines process raw reads to identify enriched epitopes, with statistical thresholds (e.g., 10 standard deviations above background) ensuring robust hit detection. The method's sensitivity enables epitope identification even at frequencies as low as 1:10,000.  

### Assessing Cellular Immunity to Specific Antigens  

The invention facilitates the assessment of cellular immunity by screening CTLs against epitope libraries derived from pathogens, tumors, or autoantigens. This enables the identification of immunodominant epitopes, cross-reactive TCR specificities, and novel antigenic targets for immunotherapy development.  

### Further Applications  

Applications of the invention include:  
- **Cancer Immunotherapy**: Identifying neoantigens for personalized vaccines or adoptive T-cell therapies.  
- **Infectious Disease**: Mapping pathogen-specific epitopes for vaccine design.  
- **Autoimmunity**: Characterizing self-reactive TCRs to develop tolerogenic strategies.  
- **Transplantation**: Screening for alloreactive T-cell epitopes to mitigate graft rejection.  

### Kits  

The invention also provides kits for implementing the method, comprising:  
- Reporter cell lines stably expressing the FRET reporter.  
- Lentiviral vectors for minigene library construction.  
- Protocols for CTL expansion, co-culture, and FACS isolation.  
- Sequencing primers and bioinformatics pipelines for data analysis.  

## Example 1  

### Model Assays Based on Stably Transfected Mouse Lymphoblastic Cell Lines EL4 and EG7 as Model Reporter Cells  

In this example, the murine T-cell lymphoma line EL4 and its ovalbumin-expressing derivative EG7 were used as reporter cells. EL4 cells were transduced with lentiviral constructs encoding the FRET reporter and minigenes containing the ovalbumin-derived epitope SIINFEKL or a scrambled control. Co-culture with OT-I TCR transgenic CTLs resulted in significant FRET signal shift in SIINFEKL-expressing cells (p < 0.0001), demonstrating epitope-specific detection. The assay was further validated using pmel-1 TCR CTLs and the hgp100 epitope KVPRNQDWL, confirming the method's robustness across different TCR/pMHC pairs.  

## Example 2  

### Confirming Function of Granzyme B-Sensitive Signal Generation Product  

The GZMB-sensitive FRET reporter was validated by co-culturing ID8 reporter cells expressing SIINFEKL or scrambled minigenes with OT-I CTLs. Flow cytometry revealed a significant increase in free CFP emission in SIINFEKL-expressing cells upon CTL recognition, confirming GZMB-mediated reporter cleavage. Specificity was further demonstrated in mixed populations, where >95% of shifted cells harbored the SIINFEKL minigene (p < 0.0001), with no bystander activation observed.  

## Example 3  

### Model Assays Based On Stably Transfected Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells  

ID8 ovarian cancer cells and their ovalbumin-expressing counterpart ID8.G7-Ova were transduced with the FRET reporter and minigene constructs. Co-culture with OT-I CTLs yielded robust FRET shift in ID8.G7-Ova cells, with signal detectable within 1 hour and peaking at 6–8 hours. Propidium iodide staining confirmed apoptosis initiation 2–4 hours post peak FRET shift, defining a safe window for target cell isolation.  

## Example 4  

### Model Assays Based on Lentivirus-Transduced Mouse Ovarian Cell Lines ID8 and ID8.G7-Ova as Reporter Cells  

Lentiviral transduction of ID8 cells with random minigene libraries enabled large-scale epitope screening. Spiking SIINFEKL minigene-expressing cells at frequencies as low as 1:10,000 allowed for epitope detection >10σ above background. Sequencing of shifted cells confirmed enrichment of the SIINFEKL minigene, validating the method's sensitivity.  

## Example 5  

### Assay Based on Autologous B-Lymphoblastoid Cell Line (B-LCL) as Reporter Cells  

Human B-lymphoblastoid cell lines (B-LCLs) were engineered to express the FRET reporter and minigene libraries. Co-culture with tumor-infiltrating lymphocytes (TILs) from vaccinated mice enabled detection of ovalbumin-specific reactivity, demonstrating the method's applicability to human systems and polyclonal T-cell populations.  

## Definitions  

- **FRET**: Förster resonance energy transfer, a fluorescence-based reporter system.  
- **GZMB**: Granzyme B, a protease delivered by cytotoxic T-cells to induce apoptosis.  
- **Minigene**: A nucleic acid construct encoding a peptide for MHC presentation.  
- **CTL**: Cytotoxic T-lymphocyte, a T-cell subset capable of inducing target cell death.  
- **APC**: Antigen-presenting cell, a cell capable of processing and presenting antigens to T-cells.  

---  

This patent application provides a comprehensive description of the invention, including detailed examples and definitions, while adhering to the specified outline and formal patent language. Let me know if you'd like any modifications or additional details.