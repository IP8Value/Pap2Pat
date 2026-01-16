Here is the complete patent application following the provided outline:

# DESCRIPTION  

## PRIORITY DATA  

This application claims priority to U.S. Provisional Patent Application No. [insert number], filed on [insert date], which is incorporated herein by reference in its entirety.  

## BACKGROUND OF THE INVENTION  

Cell-penetrating peptides (CPPs) have demonstrated unique capabilities for translocating across cellular membranes and entering mammalian cells via non-endocytic pathways. Among these, the Model Amphipathic Peptide (MAP), with the amino acid sequence KLALKLALKALKAALKLA-NH2, has been extensively studied for its ability to adopt an alpha-helical conformation. This structural arrangement positions hydrophobic side chains along one hemicircumference of the helix and positively charged side chains along the opposite side, facilitating membrane interaction and cellular uptake.  

Initial studies suggested that MAP uptake occurs through a non-endocytic mechanism, as evidenced by its internalization at low temperatures and under energy-depleted conditions. However, subsequent investigations have revealed a more complex uptake process involving both energy-dependent and independent pathways. Approximately 50% of cell-associated MAP remains membrane-bound, 30% inserts into the membrane, and 20% becomes fully internalized. Further studies using giant lipid vesicles, which lack endocytic machinery, confirmed that MAP can indeed penetrate membranes without endocytosis.  

MAP and similar CPPs have been explored for their potential to deliver therapeutic and diagnostic cargo, including proteins, nucleic acids, and small molecules, across the plasma membrane. For instance, MAP conjugated to polylysine has been used to form multiplexes with siRNA, demonstrating superior efficacy in gene silencing compared to conventional transfection methods. Additionally, MAP linked to peptide nucleic acids (PNAs) via disulfide bonds has shown successful intracellular delivery, with enhanced nuclear uptake upon endosomal release.  

Despite these advances, the relationship between cellular redox state and CPP-mediated cargo delivery remains poorly understood. Cellular redox status, governed by the balance of reduced (GSH) and oxidized (GSSG) glutathione, plays a critical role in various physiological and pathological processes. The ability to monitor and exploit redox-dependent changes in CPP uptake and cargo release could significantly enhance targeted delivery strategies.  

## DETAILED DESCRIPTION  

### Definitions  

As used herein, the following terms shall have the meanings ascribed below:  

- **Reductide**: A disulfide-linked construct comprising a cell-penetrating peptide (MAP) conjugated to a fluorescent reporter (TAMRA) and a non-cell-penetrating peptide (CLKANL) conjugated to a second fluorescent reporter (FAM), wherein the disulfide bond between the two moieties enables redox-dependent fluorescence resonance energy transfer (FRET).  
- **GSH/GSSG ratio**: The molar ratio of reduced glutathione (GSH) to oxidized glutathione (GSSG), which reflects the cellular redox state.  
- **roGFP**: Redox-sensitive green fluorescent protein, a biosensor used to measure intracellular redox potential.  

### Invention  

The present invention provides a novel disulfide-linked CPP construct, termed "reductide," designed to detect and respond to changes in cellular redox state. Reductide comprises:  
1. A MAP peptide conjugated to 5(6)-carboxytetramethylrhodamine (TAMRA) at its N-terminus.  
2. A non-cell-penetrating peptide (CLKANL) conjugated to fluorescein amidite (FAM) at its N-terminus.  
3. A disulfide bond linking the two peptides, enabling redox-dependent separation and fluorescence activation.  

Upon cellular internalization, the disulfide bond is reduced in a redox-dependent manner, releasing the FAM-labeled peptide and generating a detectable fluorescence signal. This system allows for real-time monitoring of cellular redox state and can be adapted for redox-specific delivery of therapeutic or diagnostic agents.  

### Materials and Methods  

#### Peptide Synthesis and Labeling  
Reductide was synthesized using standard FMOC solid-phase chemistry. The MAP peptide (Cys-Lys-Leu-Ala-Leu-Lys-Leu-Ala-Leu-Lys-Ala-Leu-Lys-Ala-Ala-Leu-Lys-Leu-Ala-amide) was conjugated to TAMRA via the N-terminus, while the CLKANL peptide was conjugated to FAM. The two peptides were joined via a disulfide bond, purified by HPLC, and validated by mass spectrometry.  

#### Cell Culture and Transfection  
Human fibroblasts (BJ, IMR90) and rat cardiomyocytes (H9c2) were cultured in DMEM supplemented with 10% FBS. Stable expression of Grx1-roGFP in H9c2 cells was achieved via retroviral transduction, followed by puromycin selection.  

#### Reductide Assay  
Reductide was dissolved in TBS buffer containing varying concentrations of GSH and GSSG. Fluorescence was measured using a plate reader (excitation/emission: 485 nm/528 nm for FAM; 530 nm/590 nm for TAMRA).  

#### Fluorescence Microscopy and Flow Cytometry  
Live-cell imaging was performed using an Olympus FV1000 microscope. Flow cytometry was used to quantify TAMRA and FAM fluorescence in cells pretreated with redox-modifying agents (e.g., NAC, H2O2).  

### Statistical Analysis  
Data were analyzed using Student’s t-test or ANOVA, with p < 0.05 considered significant.  

## Results  

### Effects of GSH/GSSG on Reductide Redox-Dependent Fluorescence  
In buffer containing GSH, FAM fluorescence increased over time, reflecting disulfide reduction. The addition of GSSG slowed fluorescence development and reduced maximal signal, demonstrating dependence on the GSH/GSSG ratio. TAMRA fluorescence, in contrast, was time-independent, confirming its role as a non-redox-sensitive uptake marker.  

### Flow Cytometry  
Flow cytometry revealed time-dependent increases in both TAMRA and FAM signals. NAC pretreatment enhanced TAMRA signal, indicating improved uptake under reducing conditions. FAM signal was weaker and less time-resolved, consistent with extracellular export of the reduced CLKANL peptide.  

### Reductide Response to a Small Library of Redox Modifying Compounds  
Screening with 84 redox-modifying compounds showed that 77.4% increased FAM signal (reductive shift), while 10.7% decreased signal (oxidative shift). Notably, some antioxidants exhibited pro-oxidant effects at high concentrations, highlighting the dynamic nature of redox responses.  

## Discussion  

### Reductide Uptake as Well as Reduction Depends on Cellular Redox State  
The rate of FAM signal development reflects both cellular uptake and disulfide reduction. Experiments where redox modification followed reductide incubation showed attenuated signal differences, confirming that redox state affects uptake. This property enables redox-specific targeting of therapeutic agents.  

### Pro-Oxidants Activate an Antioxidative Response  
Low-dose H2O2 pretreatment increased FAM signal, suggesting activation of antioxidative defenses. Higher doses suppressed signal, indicating oxidative stress. This biphasic response aligns with known redox-sensitive transcriptional regulation (e.g., Nrf2 activation).  

## FIGURE LEGENDS  

**Figure 1**: Reductide fluorescence in GSH/GSSG buffer. (A) FAM signal increases with GSH concentration. (B) GSSG slows signal development. (C-D) TAMRA signal is GSH-dependent but time-independent.  

**Figure 2**: Live-cell imaging of reductide distribution. TAMRA (red) localizes to the cytosol, while FAM (green) enters the nucleus.  

**Figure 3**: Correlation between reductide and roGFP signals. FAM emission correlates with roGFP ratios following H2O2 but not NAC pretreatment.  

**Figure 4**: Dose-dependent effects of redox modifiers on FAM signal. NAC enhances, while CDNB or high-dose H2O2 suppresses, fluorescence.  

**Figure 5**: Glutathione depletion (BSO) or oxidation (diamide) reduces FAM signal.  

**Figure 6**: Redox modification after reductide incubation attenuates signal differences.  

**Figure 7**: Reductide outperforms monochlorobimane in detecting redox changes.  

**Figure 8**: Flow cytometry of TAMRA and FAM signals over time.  

**Figure 9**: Redox library screening identifies compounds that modulate FAM signal.  

---  
This application provides the first redox-sensitive CPP construct capable of simultaneously reporting cellular uptake and redox state, with broad applications in drug delivery and diagnostics.