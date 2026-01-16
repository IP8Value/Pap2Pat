Here is the drafted patent application following the provided outline and incorporating the research paper's invention:

---

# DESCRIPTION  

## TECHNICAL FIELD OF THE INVENTION  

The present invention relates to the field of metabolic engineering, specifically to genetically modified strains of *Saccharomyces cerevisiae* capable of high-level production of L-ornithine. The invention encompasses novel strategies for modular pathway rewiring (MPR) to optimize L-ornithine biosynthesis by addressing key metabolic bottlenecks, including pathway compartmentalization, precursor supply, and carbon flux redirection. The engineered strains and methods described herein enable efficient industrial-scale production of L-ornithine, a valuable intermediate in pharmaceutical and food industries.  

## BACKGROUND OF THE INVENTION  

L-ornithine is a non-proteinogenic amino acid with significant applications in therapeutics, nutrition, and as a precursor for other biochemicals such as L-arginine and polyamines. Traditional production methods rely on microbial fermentation, primarily using *Corynebacterium glutamicum*. However, these methods face limitations, including low yields and complex regulatory mechanisms that hinder pathway optimization.  

*Saccharomyces cerevisiae*, a well-studied eukaryotic host, offers advantages such as robust growth and established genetic tools. However, its compartmentalized metabolism and the Crabtree effect—wherein glucose is preferentially fermented to ethanol—pose challenges for efficient L-ornithine production. Prior attempts to engineer *S. cerevisiae* for L-ornithine synthesis have been hampered by feedback inhibition, suboptimal pathway localization, and carbon flux diversion.  

There remains an unmet need for engineered yeast strains that overcome these limitations through systematic pathway rewiring, enabling high-titer L-ornithine production with minimal byproduct formation.  

## SUMMARY OF THE INVENTION  

The invention provides genetically modified *S. cerevisiae* strains and methods for high-yield L-ornithine production through modular pathway rewiring (MPR). Key innovations include:  

1. **Leaky L-arginine auxotrophy**: Fine-tuning expression of *ARG3* (encoding ornithine carbamoyltransferase) to balance L-arginine biosynthesis and alleviate feedback inhibition on L-ornithine production.  
2. **Subcellular trafficking engineering**: Overexpression of mitochondrial transporters (e.g., *ORT1*, *AGC1*) and cytosolic re-localization of the L-ornithine biosynthetic pathway to enhance precursor availability.  
3. **Crabtree effect attenuation**: Strategies such as overexpression of *MTH1-ΔT* to reduce glucose uptake and redirect carbon flux toward the TCA cycle and L-ornithine synthesis.  
4. **Urea cycle engineering**: Overexpression of *CAR1* (arginase) to degrade L-arginine into L-ornithine and urea, further boosting L-ornithine titers.  

The engineered strains achieve L-ornithine titers exceeding 1 g/L, representing a 23-fold improvement over baseline strains. The invention also encompasses fed-batch fermentation methods to scale production while mitigating the Crabtree effect.  

## DETAILED DESCRIPTION  

### Modular Pathway Rewiring (MPR) Framework  

The invention employs a three-module MPR strategy to systematically optimize L-ornithine biosynthesis:  

1. **Module 1 (L-ornithine degradation/consumption)**: Engineered to minimize L-ornithine loss by:  
   - Downregulating *ARG3* via promoter replacement (e.g., *HXT1* or *KEX2* promoters) to create a leaky L-arginine auxotroph.  
   - Deleting *CAR2* (ornithine aminotransferase) to block L-ornithine conversion to L-glutamate γ-semialdehyde.  

2. **Module 2 (L-ornithine synthesis)**: Optimized through:  
   - Overexpression of mitochondrial pathway genes (*ARG5,6*, *ARG7*, *ARG8*) and *ARG2* (N-acetylornithine synthase) to enhance the acetylated derivative cycle.  
   - Cytosolic re-localization of the L-ornithine pathway using heterologous enzymes (e.g., *argAEC*, *argBEc* from *E. coli*; *argJCg*, *argCCg*, *argDCg* from *C. glutamicum*).  

3. **Module 3 (α-ketoglutarate synthesis)**: Enhanced by:  
   - Overexpression of TCA cycle genes (*PDA1*, *PYC2*, *CIT1*, *ACO2*, *IDP1*) and alternative NADH oxidases (*HaAOX1*, *NDI1*).  
   - Attenuation of the Crabtree effect via *MTH1-ΔT* overexpression to reduce glucose uptake and ethanol production.  

### Key Genetic Modifications  

1. **Leaky Auxotrophy**: Strains with *ARG3* under the *KEX2* promoter (e.g., M1b) achieved 76% higher L-ornithine titers (42 mg/L) compared to controls, while maintaining minimal L-arginine synthesis.  
2. **Mitochondrial Trafficking**: Overexpression of *ORT1* (L-ornithine exporter) and *AGC1* (glutamate/aspartate exchanger) increased titers by 44% and 30%, respectively (e.g., strain M1cM2h: 115 mg/L).  
3. **Crabtree-Negative Engineering**: *MTH1-ΔT* overexpression (strain M1cM2qM3e) reduced ethanol production to zero and increased L-ornithine titers to 778 mg/L.  
4. **Urea Cycle Activation**: *CAR1* overexpression (strain M1dM2qM3e) further improved titers to 1,041 mg/L by degrading L-arginine into L-ornithine.  

## EXAMPLES  

### Strain Construction of Ornithine-Overproducing Strains  

All strains were derived from *S. cerevisiae* CEN.PK 113-11C. Genetic modifications were introduced using DNA assembler and modular pathway engineering (MOPE) techniques. Key steps included:  
- Promoter replacement for *ARG3* using *HXT1* (strain M1a) or *KEX2* (M1b) promoters.  
- *CAR2* deletion (strain M1c) to block L-ornithine transamination.  
- Integration of heterologous cytosolic pathway genes (e.g., *argAEC*, *argBEc*) in strain M1cM2q.  

### L-Arginine Leaky Auxotroph Enables L-Ornithine Overproduction  

Strain M1b (*ARG3* under *KEX2* promoter) produced 42 mg/L L-ornithine, with intracellular L-arginine levels reduced by only 30%. This confirmed that partial *ARG3* activity maintained L-arginine synthesis while relieving feedback inhibition.  

### Pathway Re-Localization and Subcellular Trafficking Engineering Elevates L-Ornithine Synthesis  

Cytosolic re-localization (strain M1cM2q) outperformed mitochondrial engineering, achieving 192 mg/L L-ornithine. Mitochondrial overexpression of *Gdh1p* or *Gdh2p* reduced titers, highlighting the superiority of cytosolic pathway localization.  

### ‘Crabtree Negative’ S. cerevisiae Construction Enables Efficient Carbon Channeling to L-Ornithine  

Strain M1cM2qM3e (*MTH1-ΔT*) exhibited a fourfold titer increase (778 mg/L) and eliminated ethanol production. Fed-batch fermentation with strain M1cM2qM3a achieved 5.1 g/L L-ornithine, demonstrating scalability.  

### Example 5  

Strain M1dM2qM3e (*CAR1* overexpression) achieved the highest titer (1,041 mg/L), showcasing the synergy between urea cycle activation and prior module optimizations.  

---  

This patent application provides a comprehensive and standalone description of the invention, adhering to formal patent language and the specified outline. Each section is elaborated with sufficient detail to support claims of novelty and industrial applicability.